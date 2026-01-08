import os
import threading
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM

from semsharekv import (
    LRUCacheStore, CacheItem,
    SemShareContext, set_semshare_context, disable_semshare,
    prepare_fuzzy_mapping, rearrange_past_kv,
    keep_indices_from_scores,
    PrunableDynamicCache, PrunePolicy,
    patch_llama_attention, patch_mistral_attention,
)
from semsharekv.store import pooled_cosine_01

# -------------------------
# Config
# -------------------------
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/root/autodl-tmp/modelscope-cache/models/AI-ModelScope/Mistral-7B-Instruct-v0.2",
)
OFFLINE = os.environ.get("OFFLINE", "1") == "1"
CACHE_MAX_ITEMS = int(os.environ.get("CACHE_MAX_ITEMS", "8"))
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.8"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))

# -------------------------
# App state
# -------------------------
app = FastAPI()
_lock = threading.Lock()  # 简单粗暴：串行化 GPU 推理，避免全局上下文串扰

tok = None
model = None
store = LRUCacheStore(max_items=CACHE_MAX_ITEMS)

def is_mistral_like(model_name: str) -> bool:
    n = model_name.lower()
    return ("mistral" in n) or ("mixtral" in n)

@torch.no_grad()
def get_e_cache(model, input_ids: torch.Tensor) -> torch.Tensor:
    # minimal E-cache: token embeddings
    emb = model.get_input_embeddings()(input_ids)  # [1,L,D]
    return emb.squeeze(0)

@torch.no_grad()
def legacy_tuple(past_key_values):
    pkv = past_key_values
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return tuple((pkv.key_cache[i], pkv.value_cache[i]) for i in range(len(pkv.key_cache)))
    return pkv

@torch.no_grad()
def retrieve_best_reference(tgt_e: torch.Tensor):
    best_k, best_item, best_sim = None, None, -1.0
    for k, item in store.items():
        sim = pooled_cosine_01(tgt_e, item.e_cache.to(model.device))
        if sim > best_sim:
            best_sim, best_k, best_item = sim, k, item
    return best_k, best_item, best_sim

@torch.no_grad()
def semshare_prefill(target_prompt: str, ref_item: CacheItem):
    inputs = tok(target_prompt, return_tensors="pt").to(model.device)
    tgt_ids = inputs["input_ids"]
    tgt_e = get_e_cache(model, tgt_ids)  # [Lt, D]

    ref_e = ref_item.e_cache.to(model.device)
    mapping = prepare_fuzzy_mapping(ref_e=ref_e, tgt_e=tgt_e, use_rope=True)

    ref_pkv = tuple((k.to(model.device), v.to(model.device)) for (k, v) in ref_item.past_kv)
    inj_pkv = rearrange_past_kv(ref_pkv, mapping["map"])

    ctx = SemShareContext(
        enabled=True,
        injected_past_kv=inj_pkv,
        layer_recompute_idx={},
        score_store={},
        attn_recovery=0.55,
        recompute_hot_ratio=0.5,
        recompute_cold_ratio=0.1,
        score_window=32,
    )
    set_semshare_context(ctx)
    out = model(**inputs, use_cache=True, output_attentions=False, return_dict=True)
    disable_semshare()

    pkv_tuple = legacy_tuple(out.past_key_values)
    return pkv_tuple, ctx.score_store, tgt_e, inputs

@torch.no_grad()
def build_prunable_cache(past_kv_tuple, layer_scores: dict,
                         keep_ratio=0.6, min_keep=256,
                         attn_recovery=0.55, cold_keep_ratio=0.2):
    cache = PrunableDynamicCache(config=model.config, prune_policy=PrunePolicy(keep_ratio=keep_ratio, min_keep=min_keep))
    cache.seed_from_legacy_tuple(tuple((k.detach(), v.detach()) for (k, v) in past_kv_tuple))

    for layer_idx, score in layer_scores.items():
        k = score.shape[-1]
        keep_idx = keep_indices_from_scores(
            score, attn_recovery=attn_recovery, cold_keep_ratio=cold_keep_ratio, min_keep=min_keep
        )
        keep_idx = keep_idx[:k]
        cache.set_keep_indices(layer_idx, keep_idx)
    return cache

@torch.no_grad()
def normal_prefill(prompt: str):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, use_cache=True, output_attentions=False, return_dict=True)
    e = get_e_cache(model, inputs["input_ids"]).detach().cpu()
    pkv = legacy_tuple(out.past_key_values)
    pkv_cpu = tuple((k.detach().cpu(), v.detach().cpu()) for (k, v) in pkv)
    return inputs, e, pkv_cpu

@torch.no_grad()
def generate_with_optional_semshare(prompt: str, max_new_tokens: int):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    tgt_e = get_e_cache(model, inputs["input_ids"])

    _, ref_item, sim = retrieve_best_reference(tgt_e)

    # miss -> normal generate + store
    if (ref_item is None) or (sim < SIM_THRESHOLD):
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        text = tok.decode(out_ids[0], skip_special_tokens=True)

        # 把这条也塞进 cache（可选：你也可以只缓存长 prompt）
        _, e_cpu, pkv_cpu = normal_prefill(prompt)
        store.put(f"item_{store.size()+1}", CacheItem(prompt=prompt, e_cache=e_cpu, past_kv=pkv_cpu))
        return text, sim, False

    # hit -> semshare prefill + prunable cache + generate
    past_kv_tuple, layer_scores, _, _ = semshare_prefill(prompt, ref_item)
    prunable_cache = build_prunable_cache(past_kv_tuple, layer_scores)

    out_ids = model.generate(**inputs, past_key_values=prunable_cache,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    text = tok.decode(out_ids[0], skip_special_tokens=True)

    # 写回 cache：把“目标 prompt 的真实 prefill KV”缓存起来（下次更像真正 KV cache）
    _, e_cpu, pkv_cpu = normal_prefill(prompt)
    store.put(f"item_{store.size()+1}", CacheItem(prompt=prompt, e_cache=e_cpu, past_kv=pkv_cpu))
    return text, sim, True

# -------------------------
# FastAPI schema
# -------------------------
class GenReq(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = None

class GenResp(BaseModel):
    text: str
    retrieved_sim: float
    semshare_hit: bool
    cache_items: int

@app.on_event("startup")
def _startup():
    global tok, model

    if OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"  # 避免任何 Hub HTTP 请求
    local_only = True  # 你用的是 ModelScope 本地目录，强制本地加载

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=local_only,
    )
    model.generation_config.pad_token_id = tok.eos_token_id

    if is_mistral_like(MODEL_DIR):
        patch_mistral_attention()
    else:
        patch_llama_attention()

@app.get("/healthz")
def healthz():
    return {"ok": True, "model_dir": MODEL_DIR, "cache_items": store.size()}

@app.post("/generate", response_model=GenResp)
def generate(req: GenReq):
    # 注意：Transformers + GPU 推理是阻塞的；简单起见先串行化
    with _lock, torch.no_grad():
        max_new = req.max_new_tokens or MAX_NEW_TOKENS
        text, sim, hit = generate_with_optional_semshare(req.prompt, max_new)
        return GenResp(text=text, retrieved_sim=float(sim), semshare_hit=bool(hit), cache_items=store.size())
