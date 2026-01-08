# SemanticShareKV

A lightweight **semantic KV-cache reuse + pruning** prototype for HuggingFace `transformers`, with a simple **FastAPI server** that provides an **online (in-memory) semantic cache store**.

This repo explores **reusing KV cache across semantically similar prompts** (instead of strict prefix match), to reduce prefill compute and optionally prune KV to save memory.

This repository is an independent re-implementation of the method described in [arXiv:2509.24832].
It is not an official code release and is not affiliated with the paper’s authors.
---

## What this repo does

Given an incoming prompt **P**:

1. Compute a cheap embedding representation (`E-cache`) using the model’s **input embedding layer**.
2. Search an in-memory **LRU semantic store** for the **most similar** cached prompt **R**.
3. If similarity ≥ threshold:
   - Build a **fuzzy token mapping** from `R` → `P`
   - **Rearrange** cached `past_key_values` from `R` to match `P`
   - Monkeypatch attention (Mistral/Llama) to **inject** the rearranged KV during prefill
   - Collect per-layer token importance scores (optional)
   - Build a `PrunableDynamicCache` and pass it into `generate()`
4. Store this prompt’s prefill KV into the store (online, no persistence).

> Even when “cache hit” happens, the output text may still differ because decoding can be stochastic when `temperature > 0`, and pruning/recompute can slightly change logits. For debugging determinism, set `temperature=0`.

---

## Repository layout

```
.
├── src/semsharekv/
│   ├── __init__.py
│   ├── server.py                 # FastAPI server (online semantic cache store)
│   ├── semshare_context.py        # global SemShare context
│   ├── store.py                   # LRUCacheStore, CacheItem, similarity utils
│   ├── mapping.py                 # fuzzy mapping (ref -> target)
│   ├── kv_rearrange.py            # rearrange past_kv by mapping
│   ├── cache_prune.py             # PrunableDynamicCache + prune policies
│   └── monkeypatch/
│       ├── mistral.py             # MistralAttention patch
│       └── llama.py               # LlamaAttention patch
└── examples/
    └── example_semsharekv.py      # standalone demo (local model_dir)
```

---

## Installation

### 1) Create/activate env

```bash
conda create -n semsharekv python=3.10 -y
conda activate semsharekv
```

### 2) Install deps

```bash
pip install -U pip setuptools wheel
pip install torch transformers fastapi uvicorn faiss-cpu
```

### 3) Install this repo as editable

From the repo root:

```bash
pip install -e .
```

Why `-e`?
- Installs an importable package (`semsharekv`) while keeping your source code “live”.
- After that, editing `src/semsharekv/*.py` usually **does not require reinstall**.

---

## Model setup (offline / ModelScope)

This repo expects `transformers`-style model folders on disk (with `config.json`, tokenizer files, weights).  
You can pass your ModelScope-downloaded directory directly via `--model_dir /path/to/model`.

Examples on AutoDL:

- Mistral:
  - `/root/autodl-tmp/modelscope-cache/models/AI-ModelScope/Mistral-7B-Instruct-v0.2`
- Llama:
  - `/root/autodl-tmp/modelscope-cache/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`

---

## Quickstart: run the example

### Mistral

```bash
python examples/example_semsharekv.py \
  --model_dir /root/autodl-tmp/modelscope-cache/models/AI-ModelScope/Mistral-7B-Instruct-v0.2
```

### Llama 3.1

```bash
python examples/example_semsharekv.py \
  --model_dir /root/autodl-tmp/modelscope-cache/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
```

---

## Run the server (online semantic cache store)

### Start server

```bash
python -m semsharekv.server \
  --model_dir /root/autodl-tmp/modelscope-cache/models/AI-ModelScope/Mistral-7B-Instruct-v0.2 \
  --port 8008 \
  --max_items 64
```

You should see:

- `Uvicorn running on http://0.0.0.0:8008`

### Call `/v1/generate`

> Tip: You don’t need `jq`. Use `python -m json.tool`.

```bash
curl -sS http://127.0.0.1:8008/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Summarize: Apple posted earnings and iPhone sales rose...","max_new_tokens":80,"use_semshare":true,"sim_threshold":0.8}' \
| python -m json.tool
```

### Cache stats / clear

```bash
curl -sS http://127.0.0.1:8008/v1/cache/stats | python -m json.tool
curl -sS -X POST http://127.0.0.1:8008/v1/cache/clear | python -m json.tool
```

---

## API

### `POST /v1/generate`

Request JSON:

```json
{
  "prompt": "string",
  "max_new_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9,
  "use_semshare": true,
  "sim_threshold": 0.8,
  "store_key": null
}
```

Response JSON (example):

```json
{
  "text": "...",
  "used_semshare": true,
  "cache_hit": true,
  "ref_key": "pXXXXXXXX",
  "retrieved_sim": 0.93,
  "store_key": "pYYYYYYYY"
}
```

---

## Notes & troubleshooting

### 1) “cache hit but output different”
Expected if:
- `temperature > 0` (sampling)
- pruning / recompute slightly changes logits

To debug determinism, set `temperature=0`:

```bash
curl -sS http://127.0.0.1:8008/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"...","max_new_tokens":80,"temperature":0,"top_p":1.0,"use_semshare":true,"sim_threshold":0.8}' \
| python -m json.tool
```

### 2) `jq: command not found`
Either install:

```bash
apt-get update && apt-get install -y jq
```

Or just use:

```bash
| python -m json.tool
```

### 3) Persistence
The server is **in-memory only**. Restarting clears the cache.  
If you later want persistence, the clean extension point is `LRUCacheStore.put/get/items()`.

---

## Disclaimer

This is a research prototype for semantic KV reuse and cache pruning.  
Not optimized for production throughput / multi-process deployment (yet).
