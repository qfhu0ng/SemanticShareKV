import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from semsharekv import (
    LRUCacheStore, CacheItem,
    prepare_fuzzy_mapping, rearrange_past_kv,
    keep_indices_from_scores,
    PrunableDynamicCache, PrunePolicy,
    SemShareContext, set_semshare_context, disable_semshare,
    patch_llama_attention, patch_mistral_attention
)
from semsharekv.store import pooled_cosine_01

# NEW: LSH store
from semsharekv.store_lsh import LSHSemanticStore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--max_items", type=int, default=4)
    p.add_argument("--store", choices=["lru", "lsh"], default="lru")
    p.add_argument("--lsh_bits", type=int, default=256)
    p.add_argument("--lsh_topk", type=int, default=8)
    p.add_argument("--sim_threshold", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=64)
    return p.parse_args()


def is_mistral_like(model_name: str) -> bool:
    n = model_name.lower()
    return ("mistral" in n) or ("mixtral" in n)


@torch.no_grad()
def get_e_cache(model, input_ids: torch.Tensor) -> torch.Tensor:
    # minimal E-cache: token embeddings (cheap)
    emb = model.get_input_embeddings()(input_ids)  # [1,L,D]
    return emb.squeeze(0)  # [L,D]


@torch.no_grad()
def prefill_reference_and_store(model, tok, store, prompt: str, key: str):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, use_cache=True, output_attentions=False, return_dict=True)
    e = get_e_cache(model, inputs["input_ids"]).detach().cpu()

    pkv = out.past_key_values
    # HF may return Cache object; convert to legacy tuple(k,v)
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        pkv_tuple = tuple(
            (pkv.key_cache[i].detach().cpu(), pkv.value_cache[i].detach().cpu())
            for i in range(len(pkv.key_cache))
        )
    else:
        pkv_tuple = tuple((k.detach().cpu(), v.detach().cpu()) for (k, v) in pkv)

    store.put(key, CacheItem(prompt=prompt, e_cache=e, past_kv=pkv_tuple))
    return key


@torch.no_grad()
def retrieve_best_reference(store, tgt_e: torch.Tensor, device: torch.device):
    """
    LRU: linear scan
    LSH: store.search_best()
    """
    if isinstance(store, LSHSemanticStore):
        ref_key, ref_item, sim, _debug = store.search_best(tgt_e, device)
        return ref_key, ref_item, sim

    best_key, best_item, best_sim = None, None, -1.0
    for k, item in store.items():
        sim = pooled_cosine_01(tgt_e, item.e_cache.to(device))
        if sim > best_sim:
            best_sim, best_key, best_item = float(sim), k, item
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

    # mapping + rearranged KV
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
def build_prunable_cache(model, past_kv_tuple, layer_scores: dict, keep_ratio=0.6, min_keep=256,
                         attn_recovery=0.55, cold_keep_ratio=0.2):
    cache = PrunableDynamicCache(config=model.config, prune_policy=PrunePolicy(keep_ratio=keep_ratio, min_keep=min_keep))

    # be defensive about None
    legacy = []
    for kv in past_kv_tuple:
        if kv is None:
            continue
        k, v = kv
        if k is None or v is None:
            continue
        legacy.append((k.detach(), v.detach()))
    cache.seed_from_legacy_tuple(tuple(legacy))

    for layer_idx, score in layer_scores.items():
        if score is None:
            continue
        keep_idx = keep_indices_from_scores(
            score,
            attn_recovery=attn_recovery,
            cold_keep_ratio=cold_keep_ratio,
            min_keep=min_keep
        )
        cache.set_keep_indices(layer_idx, keep_idx)

    return cache


@torch.no_grad()
def semshare_generate(model, tok, store, target_prompt: str, sim_threshold=0.8, max_new_tokens=64):
    inputs = tok(target_prompt, return_tensors="pt").to(model.device)
    tgt_e = get_e_cache(model, inputs["input_ids"])

    ref_key, ref_item, sim = retrieve_best_reference(store, tgt_e, model.device)

    if (ref_item is None) or (sim is None) or (sim < sim_threshold):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return tok.decode(out[0], skip_special_tokens=True), sim, False, ref_key

    past_kv_tuple, layer_scores = semshare_prefill(model, tok, target_prompt, ref_item)
    prunable_cache = build_prunable_cache(model, past_kv_tuple, layer_scores)

    out = model.generate(**inputs, past_key_values=prunable_cache, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True), sim, True, ref_key


def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    # patch by model family
    if is_mistral_like(args.model_dir):
        patch_mistral_attention()
    else:
        patch_llama_attention()

    # NEW: choose store
    if args.store == "lsh":
        store = LSHSemanticStore(
            max_items=args.max_items,
            dim=int(model.config.hidden_size),
            nbits=args.lsh_bits,
            topk=args.lsh_topk,
        )
    else:
        store = LRUCacheStore(max_items=args.max_items)

    # build a reference entry
    ref_prompt = (
        "Summarize the following articles into 5 bullet points.\n"
        "Article A: ... (put a long multi-paragraph text here) ...\n"
        "Article B: ... (another long text) ...\n"
    )
    prefill_reference_and_store(model, tok, store, ref_prompt, key="ref1")

    # semantically similar target (paraphrased)
    tgt_prompt = (
        "Please write a concise 5-bullet summary of these news pieces.\n"
        "Text 1: ... (paraphrased but similar long text) ...\n"
        "Text 2: ... (paraphrased) ...\n"
    )

    ans, sim, hit, ref_key = semshare_generate(
        model, tok, store,
        tgt_prompt,
        sim_threshold=args.sim_threshold,
        max_new_tokens=args.max_new_tokens
    )
    print("store =", args.store)
    print("retrieved_sim =", sim)
    print("cache_hit =", hit, "ref_key =", ref_key)
    print(ans)


if __name__ == "__main__":
    main()
