# SemanticShareKV (Reproduction of arXiv:2509.24832)

> **Status:** research prototype / reproduction  
> This repository is an **independent re-implementation** inspired by **SemShareKV** (arXiv:2509.24832).  
> It focuses on **semantic KV-cache reuse across prompts** and **token-level KV pruning** during generation.

---

## 1. Background (What problem this solves)

When you run a decoder-only LLM with long prompts, the **prefill** phase is expensive and the **KV cache** grows quickly.  
Most KV optimizations either:

- compress the KV cache *within one prompt*, or
- reuse KV only when there is an *exact prefix match*.

**SemShareKV** targets the more common case where prompts are **semantically similar but lexically different**, and tries to:

- **reuse** a previously computed prompt’s `past_key_values` for a new prompt (**semantic cache hit**), and
- **prune** unimportant cached tokens to reduce memory during decoding.

---

## 2.  Repository layout 

```
src/semsharekv/
  __init__.py
  server.py            # FastAPI server + demo pipeline
  store_lsh.py         # CacheItem + LRU store (+ experimental LSH store)
  lsh.py               # token-level LSH matching (FAISS IndexLSH) + sim score
  matching.py          # fuzzy mapping + KV rearrangement
  semshare_context.py  # runtime context shared by monkeypatch
  selection.py         # keep_indices_from_scores (hot+cool token selection)
  prunable_cache.py    # PrunableDynamicCache (HF DynamicCache subclass)
  rope.py              # optional RoPE-style embedding transform for matching
  monkeypatch/
    llama.py           # patch HF LlamaAttention (transformers >= 4.56)
    mistral.py         # patch HF MistralAttention (transformers >= 4.56)
```

---

## 3. Installation

### 3.1 Create environment

```bash
conda create -n semsharekv python=3.10 -y
conda activate semsharekv
```

### 3.2 Install dependencies

Core deps (package requirements):

```bash
pip install -U pip setuptools wheel
pip install torch transformers faiss-cpu
```

Server deps (required if you run `semsharekv.server`):

```bash
pip install fastapi uvicorn pydantic
```

### 3.3 Install this repo (editable)

From repo root:

```bash
pip install -e .
```

---

## 4. Model setup (offline friendly)

This repo assumes a local HuggingFace-style model directory (e.g., from ModelScope / HF cache) that contains:

- `config.json`, tokenizer files, and weights

When running the server we set:

- `HF_HOME` (optional, defaults to `/root/autodl-tmp/hf-cache`)
- `HF_HUB_OFFLINE=1` when `--offline` is enabled

---

## 5. Quickstart: run the FastAPI server

### 5.1 Start server

```bash
python -m semsharekv.server \
  --model_dir /path/to/your/model \
  --host 0.0.0.0 \
  --port 8008 \
  --max_items 64 \
  --offline
```

You should see:

- `Uvicorn running on http://0.0.0.0:8008`

### 5.2 Generate (with semantic reuse enabled)

```bash
curl -sS http://127.0.0.1:8008/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Summarize the following article: ...",
    "max_new_tokens": 120,
    "temperature": 0.0,
    "top_p": 1.0,
    "use_semshare": true,
    "sim_threshold": 0.80
  }' | python -m json.tool
```

Example response fields:

```json
{
  "text": "...",
  "used_semshare": true,
  "cache_hit": true,
  "ref_key": "p1234abcd",
  "retrieved_sim": 0.93,
  "store_key": "p9f00cafe",
  "start_ms": 1767...,
  "latency_ms": 531.2,
  "run_key": "H_531ms_1767..._p9f00cafe"
}
```

### 5.3 Cache stats / clear

```bash
curl -sS http://127.0.0.1:8008/v1/cache/stats | python -m json.tool
curl -sS -X POST http://127.0.0.1:8008/v1/cache/clear | python -m json.tool
```

---

## 6. Configuration knobs (practical tuning)

### Retrieval

- `sim_threshold`: higher → fewer cache hits but safer quality
- LSH params (see `lsh.py`):
  - `nbits`: more bits → higher resolution, typically slower / more memory

### Recompute / prune

In `server.py -> SemShareContext(...)` and `build_prunable_cache(...)`:

- `attn_recovery`: how much attention mass to recover with “hot” tokens
- `recompute_hot_ratio`, `recompute_cold_ratio`: (experimental) ratios for recompute selection
- `keep_ratio`, `min_keep`: pruning budget in `PrunePolicy`
- `cold_keep_ratio`: extra cold-token budget beyond hot tokens

---

## 7. Notes / Known limitations

- **This is a prototype**: designed for readability and reproduction, not production throughput.
- **Monkeypatching is version-sensitive**: the attention patch targets newer Transformers APIs (Cache/DynamicCache).
- **No multi-process shared cache**: the cache store is in-memory within one process.
- **Similarity metric vs mapping**:
  - Retrieval similarity uses token-level LSH matching.
  - Mapping currently uses cosine top-k on CPU; swapping mapping to fully LSH-based candidate retrieval is a clean next step.

---

## 8.  Disclaimer

This repo is **not** an official release, and is **not affiliated** with the paper’s authors.  
It is intended for research, debugging, and reproducibility experiments.