# src/semsharekv/server.py
import os
import argparse
import asyncio
import hashlib
from typing import Optional, Dict, Any, Tuple

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from semsharekv import (
    LRUCacheStore, CacheItem,
    prepare_fuzzy_mapping, rearrange_past_kv,
    keep_indices_from_scores,
    PrunableDynamicCache, PrunePolicy,
    SemShareContext, set_semshare_context, disable_semshare,
    patch_llama_attention, patch_mistral_attention,
)
from semsharekv.store import pooled_cosine_01


def is_mistral_like(name: str) -> bool:
    n = name.lower()
    return ("mistral" in n) or ("mixtral" in n)


@torch.no_grad()
def get_e_cache(model, input_ids: torch.Tensor) -> torch.Tensor:
    # Minimal E-cache: token embeddings [L, D]
    emb = model.get_input_embeddings()(input_ids)  # [1, L, D]
    return emb.squeeze(0)  # [L, D]


@torch.no_grad()
def prefill_and_make_item(model, tok, prompt: str) -> CacheItem:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, use_cache=True, output_attentions=False, return_dict=True)

    e = get_e_cache(model, inputs["input_ids"]).detach().cpu()
    pkv = out.past_key_values

    # HF may return Cache object; convert to legacy tuple(k,v) per layer
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        pkv_tuple = tuple(
            (pkv.key_cache[i].detach().cpu(), pkv.value_cache[i].detach().cpu())
            for i in range(len(pkv.key_cache))
        )
    else:
        pkv_tuple = tuple((k.detach().cpu(), v.detach().cpu()) for (k, v) in pkv)

    return CacheItem(prompt=prompt, e_cache=e, past_kv=pkv_tuple)


@torch.no_grad()
def retrieve_best_reference(
    store: LRUCacheStore, tgt_e: torch.Tensor, device: torch.device
) -> Tuple[Optional[str], Optional[CacheItem], float]:
    best_key, best_item, best_sim = None, None, -1.0
    for k, item in store.items():
        sim = pooled_cosine_01(tgt_e, item.e_cache.to(device))
        if sim > best_sim:
            best_sim, best_key, best_item = sim, k, item
    return best_key, best_item, best_sim


@torch.no_grad()
def semshare_prefill(model, tok, target_prompt: str, ref_item: CacheItem):
    inputs = tok(target_prompt, return_tensors="pt").to(model.device)
    tgt_ids = inputs["input_ids"]
    tgt_e = get_e_cache(model, tgt_ids)  # [Lt, D]

    ref_e = ref_item.e_cache.to(model.device)
    mapping = prepare_fuzzy_mapping(ref_e=ref_e, tgt_e=tgt_e, use_rope=True)

    ref_pkv = tuple((k.to(model.device), v.to(model.device)) for (k, v) in ref_item.past_kv)
    inj_pkv = rearrange_past_kv(ref_pkv, mapping["map"])

    ctx = SemShareContext(
        enabled=True,
        injected_past_kv=inj_pkv,     # injected KV for each layer
        layer_recompute_idx={},       # let monkeypatch fill
        score_store={},               # will be filled by monkeypatch
        attn_recovery=0.55,
        recompute_hot_ratio=0.5,
        recompute_cold_ratio=0.1,
        score_window=32,
    )
    set_semshare_context(ctx)

    out = model(**inputs, use_cache=True, output_attentions=False, return_dict=True)

    disable_semshare()  # IMPORTANT: decode uses HF cache normally

    pkv = out.past_key_values
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        pkv_tuple = tuple((pkv.key_cache[i], pkv.value_cache[i]) for i in range(len(pkv.key_cache)))
    else:
        pkv_tuple = pkv

    return pkv_tuple, ctx.score_store


@torch.no_grad()
def build_prunable_cache(
    model,
    past_kv_tuple,
    layer_scores: dict,
    keep_ratio=0.6,
    min_keep=256,
    attn_recovery=0.55,
    cold_keep_ratio=0.2,
):
    cache = PrunableDynamicCache(
        config=model.config,
        prune_policy=PrunePolicy(keep_ratio=keep_ratio, min_keep=min_keep),
    )

    # Convert legacy tuple; skip None layers defensively
    legacy = []
    for kv in past_kv_tuple:
        if kv is None:
            legacy.append(None)
            continue
        k, v = kv
        if k is None or v is None:
            legacy.append(None)
            continue
        legacy.append((k.detach(), v.detach()))

    # seed_from_legacy_tuple expects a tuple of (k,v) for each layer;
    # if you used None placeholders, filter them out and rely on set_keep_indices for valid layers.
    compact = tuple([x for x in legacy if x is not None])
    cache.seed_from_legacy_tuple(compact)

    for layer_idx, score in layer_scores.items():
        if score is None:
            continue
        keep_idx = keep_indices_from_scores(
            score,
            attn_recovery=attn_recovery,
            cold_keep_ratio=cold_keep_ratio,
            min_keep=min_keep,
        )
        cache.set_keep_indices(layer_idx, keep_idx)

    return cache


class GenerateReq(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    use_semshare: bool = True
    sim_threshold: float = 0.8
    store_key: Optional[str] = None  # if provided, store under this key; else auto


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--max_items", type=int, default=64)
    p.add_argument("--hf_home", type=str, default=os.environ.get("HF_HOME", "/root/autodl-tmp/hf-cache"))
    # default offline (no HF network). use --online to allow network.
    p.add_argument("--online", action="store_true", help="Allow HF Hub network access (default: offline)")
    return p.parse_args()


def create_app(model, tok, store: LRUCacheStore):
    app = FastAPI()
    gpu_lock = asyncio.Lock()  # serialize GPU calls for stability

    @app.get("/v1/cache/stats")
    async def cache_stats() -> Dict[str, Any]:
        keys = [k for k, _ in store.items()]
        return {"items": len(keys), "keys": keys}

    @app.post("/v1/cache/clear")
    async def cache_clear() -> Dict[str, Any]:
        store.clear()
        return {"ok": True}

    @app.post("/v1/generate")
    async def generate(req: GenerateReq) -> Dict[str, Any]:
        async with gpu_lock:
            inputs = tok(req.prompt, return_tensors="pt").to(model.device)
            tgt_e = get_e_cache(model, inputs["input_ids"])

            sim: Optional[float] = None
            used_semshare = False
            cache_hit = False
            ref_key: Optional[str] = None

            # ---- generation ----
            if req.use_semshare:
                # only try retrieve if store not empty
                any_item = False
                for _k, _v in store.items():
                    any_item = True
                    break

                if any_item:
                    ref_key, ref_item, sim = retrieve_best_reference(store, tgt_e, model.device)
                    if (ref_item is not None) and (sim is not None) and (sim >= req.sim_threshold):
                        past_kv_tuple, layer_scores = semshare_prefill(model, tok, req.prompt, ref_item)
                        prunable_cache = build_prunable_cache(model, past_kv_tuple, layer_scores)
                        out = model.generate(
                            **inputs,
                            past_key_values=prunable_cache,
                            max_new_tokens=req.max_new_tokens,
                            do_sample=req.temperature > 0,
                            temperature=req.temperature,
                            top_p=req.top_p,
                        )
                        used_semshare = True
                        cache_hit = True
                    else:
                        out = model.generate(
                            **inputs,
                            max_new_tokens=req.max_new_tokens,
                            do_sample=req.temperature > 0,
                            temperature=req.temperature,
                            top_p=req.top_p,
                        )
                else:
                    out = model.generate(
                        **inputs,
                        max_new_tokens=req.max_new_tokens,
                        do_sample=req.temperature > 0,
                        temperature=req.temperature,
                        top_p=req.top_p,
                    )
            else:
                out = model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    do_sample=req.temperature > 0,
                    temperature=req.temperature,
                    top_p=req.top_p,
                )

            text = tok.decode(out[0], skip_special_tokens=True)

            # ---- store (online store, no persistence) ----
            # stable key: md5(prompt)[:8]
            key = req.store_key or ("p" + hashlib.md5(req.prompt.encode("utf-8")).hexdigest()[:8])
            item = prefill_and_make_item(model, tok, req.prompt)
            store.put(key, item)

            return {
                "text": text,
                "used_semshare": used_semshare,
                "cache_hit": cache_hit,
                "ref_key": ref_key,
                "retrieved_sim": float(sim) if sim is not None else None,
                "store_key": key,
            }

    return app


def main():
    args = parse_args()

    os.environ["HF_HOME"] = args.hf_home
    os.environ.pop("TRANSFORMERS_CACHE", None)
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )

    if is_mistral_like(args.model_dir):
        patch_mistral_attention()
    else:
        patch_llama_attention()

    store = LRUCacheStore(max_items=args.max_items)
    app = create_app(model, tok, store)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
