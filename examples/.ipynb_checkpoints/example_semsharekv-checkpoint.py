import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from semsharekv import (
    LRUCacheStore, CacheItem,
    prepare_fuzzy_mapping, rearrange_past_kv,
    keep_indices_from_scores,
    PrunableDynamicCache, PrunePolicy,
    SemShareContext, set_semshare_context, disable_semshare,
    patch_llama_attention, patch_mistral_attention
)
from semsharekv.store import pooled_cosine_01


def parse_args():
    p = argparse.ArgumentParser(description="SemShareKV demo (local model dir via --model_dir)")
    p.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("MODEL_DIR", ""),
        help="Local model directory (e.g. ModelScope cache dir). If empty, fallback to a HF repo id.",
    )
    p.add_argument(
        "--hf_home",
        type=str,
        default=os.environ.get("HF_HOME", "/root/autodl-tmp/hf-cache"),
        help="HF cache root dir (optional).",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Force offline mode (HF_HUB_OFFLINE=1). Recommended for ModelScope local dirs.",
    )
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--sim_threshold", type=float, default=0.8)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device_map", type=str, default="auto")
    return p.parse_args()


def is_mistral_like(model_name: str) -> bool:
    n = model_name.lower()
    return ("mistral" in n) or ("mixtral" in n)


@torch.no_grad()
def get_e_cache(model, input_ids: torch.Tensor) -> torch.Tensor:
    # minimal E-cache: token embeddings (cheap). You can swap to contextual states later.
    emb = model.get_input_embeddings()(input_ids)  # [1,L,D]
    return emb.squeeze(0)


@torch.no_grad()
def prefill_reference_and_store(model, tok, store: LRUCacheStore, prompt: str, key: str):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, use_cache=True, output_attentions=False, return_dict=True)
    e = get_e_cache(model, inputs["input_ids"]).detach().cpu()

    pkv = out.past_key_values
    # legacy tuple expected here; if it's a Cache object, convert by reading key_cache/value_cache
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        pkv_tuple = tuple((pkv.key_cache[i].detach().cpu(), pkv.value_cache[i].detach().cpu())
                          for i in range(len(pkv.key_cache)))
    else:
        pkv_tuple = tuple((k.detach().cpu(), v.detach().cpu()) for (k, v) in pkv)

    store.put(key, CacheItem(prompt=prompt, e_cache=e, past_kv=pkv_tuple))
    return key


@torch.no_grad()
def retrieve_best_reference(store: LRUCacheStore, tgt_e: torch.Tensor, device: torch.device):
    best_key, best_item, best_sim = None, None, -1.0
    for k, item in store.items():
        sim = pooled_cosine_01(tgt_e, item.e_cache.to(device))
        if sim > best_sim:
            best_sim, best_key, best_item = sim, k, item
    return best_key, best_item, best_sim


@torch.no_grad()
def semshare_prefill(model, tok, target_prompt: str, ref_item: CacheItem):
    """
    1) compute fuzzy mapping
    2) rearrange ref KV to align with target len
    3) enable SemShareContext and run a single prefill forward to obtain SemShare past_kv (legacy tuple)
    4) return (past_kv_tuple, per_layer_scores)
    """
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

    disable_semshare()  # IMPORTANT: decode uses HF cache normally

    pkv = out.past_key_values
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        pkv_tuple = tuple((pkv.key_cache[i], pkv.value_cache[i]) for i in range(len(pkv.key_cache)))
    else:
        pkv_tuple = pkv

    return pkv_tuple, ctx.score_store


@torch.no_grad()
def build_prunable_cache(model, past_kv_tuple, layer_scores: dict,
                         keep_ratio=0.6, min_keep=256,
                         attn_recovery=0.55, cold_keep_ratio=0.2):
    cache = PrunableDynamicCache(config=model.config, prune_policy=PrunePolicy(keep_ratio=keep_ratio, min_keep=min_keep))
    cache.seed_from_legacy_tuple(tuple((k.detach(), v.detach()) for (k, v) in past_kv_tuple))

    for layer_idx, score in layer_scores.items():
        k = score.shape[-1]
        keep_idx = keep_indices_from_scores(
            score,
            attn_recovery=attn_recovery,
            cold_keep_ratio=cold_keep_ratio,
            min_keep=min_keep
        )
        keep_idx = keep_idx[:k]
        cache.set_keep_indices(layer_idx, keep_idx)

    return cache


@torch.no_grad()
def semshare_generate(model, tok, store: LRUCacheStore, target_prompt: str, sim_threshold=0.8, max_new_tokens=64):
    inputs = tok(target_prompt, return_tensors="pt").to(model.device)
    tgt_e = get_e_cache(model, inputs["input_ids"])

    _, ref_item, sim = retrieve_best_reference(store, tgt_e, model.device)
    if (ref_item is None) or (sim < sim_threshold):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0], skip_special_tokens=True), sim

    past_kv_tuple, layer_scores = semshare_prefill(model, tok, target_prompt, ref_item)

    prunable_cache = build_prunable_cache(
        model,
        past_kv_tuple,
        layer_scores,
        keep_ratio=0.6,
        min_keep=256,
        attn_recovery=0.55,
        cold_keep_ratio=0.2
    )

    out = model.generate(**inputs, past_key_values=prunable_cache,
                         max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True), sim


def main():
    args = parse_args()

    # Optional: set HF cache dir (doesn't affect ModelScope local loading, but avoids deprecation warnings elsewhere)
    os.environ["HF_HOME"] = args.hf_home

    # Offline to prevent any Hub HTTP calls
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"  # offline mode described in HF docs
        local_only = True
    else:
        local_only = bool(args.model_dir.strip())

    model_id = args.model_dir.strip() or "mistralai/Mistral-7B-Instruct-v0.2"

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=local_only)
    cfg = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    is_llama = "llama" in cfg.model_type.lower()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        attn_implementation="eager" if is_llama else None,
    )

    # silence "Setting pad_token_id to eos_token_id..."
    model.generation_config.pad_token_id = tok.eos_token_id  # avoids the generation warning by explicit pad_token_id
    # (setting pad_token_id explicitly is the standard fix)
    # see HF community guidance

    if is_mistral_like(model_id):
        patch_mistral_attention()
    else:
        patch_llama_attention()

    store = LRUCacheStore(max_items=4)

    ref_prompt = (
        "Summarize the following articles into 5 bullet points.\n"
        "Article A: ... (put a long multi-paragraph text here) ...\n"
        "Article B: ... (another long text) ...\n"
    )
    prefill_reference_and_store(model, tok, store, ref_prompt, key="ref1")

    tgt_prompt = (
        "Please write a concise 5-bullet summary of these news pieces.\n"
        "Text 1: ... (paraphrased but similar long text) ...\n"
        "Text 2: ... (paraphrased) ...\n"
    )

    ans, sim = semshare_generate(
        model, tok, store, tgt_prompt,
        sim_threshold=args.sim_threshold,
        max_new_tokens=args.max_new_tokens
    )
    print("retrieved_sim =", sim)
    print(ans)


if __name__ == "__main__":
    main()
