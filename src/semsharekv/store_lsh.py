from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import faiss  # faiss-cpu
except Exception as e:
    faiss = None

from .store import pooled_cosine_01  # 你原来的精确 sim（0~1）


def _pool_vec(e_cache) -> np.ndarray:
    """
    e_cache: torch.Tensor [L, D] on CPU/GPU
    -> pooled vec: np.float32 [D], L2-normalized
    """
    # 避免引入 torch type hints：用 duck typing
    x = e_cache
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "float"):
        x = x.float()
    if hasattr(x, "mean"):
        x = x.mean(dim=0)  # [D]
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()

    v = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-12
    v = v / n
    return v


@dataclass
class LSHSearchDebug:
    approx_ids: List[int]
    approx_distances: List[float]


class LSHSemanticStore:
    """
    一个“在线的 semantic KV cache store”：
    - 仍然存 CacheItem（prompt, e_cache, past_kv）
    - 检索用 LSH 先召回 topK 候选
    - 再用 pooled_cosine_01 做精确重排，输出 sim（0~1）

    说明：
    - FAISS IndexLSH 不太适合频繁 delete，所以这里采用“LRU 驱逐/更新后重建索引”的策略；
      max_items=64/256/1024 这种级别完全够用。
    """

    def __init__(self, max_items: int, dim: int, nbits: int = 256, topk: int = 8):
        if faiss is None:
            raise RuntimeError("faiss is not available. Please install faiss-cpu in this env.")
        self.max_items = int(max_items)
        self.dim = int(dim)
        self.nbits = int(nbits)
        self.topk = int(topk)

        # LRU：key -> item
        self._od: "OrderedDict[str, object]" = OrderedDict()

        # key -> pooled vec (np.float32 [D])
        self._vecs: Dict[str, np.ndarray] = {}

        # internal id mapping for faiss
        self._key2id: Dict[str, int] = {}
        self._id2key: Dict[int, str] = {}
        self._next_id: int = 1

        self._index = None
        self._rebuild_index()

    def _rebuild_index(self):
        # 用 LSH 做召回（距离是 faiss 的 L2 on hashed space 的 proxy；我们只用来召回）
        base = faiss.IndexLSH(self.dim, self.nbits)
        index = faiss.IndexIDMap2(base)

        ids = []
        mat = []
        for k, v in self._vecs.items():
            idx = self._key2id.get(k)
            if idx is None:
                continue
            ids.append(idx)
            mat.append(v)

        if len(mat) > 0:
            xb = np.stack(mat, axis=0).astype(np.float32)
            index.add_with_ids(xb, np.asarray(ids, dtype=np.int64))

        self._index = index

    def __len__(self) -> int:
        return len(self._od)

    def items(self) -> Iterable[Tuple[str, object]]:
        return self._od.items()

    def clear(self):
        self._od.clear()
        self._vecs.clear()
        self._key2id.clear()
        self._id2key.clear()
        self._next_id = 1
        self._rebuild_index()

    def get(self, key: str):
        item = self._od.get(key)
        if item is None:
            return None
        # touch for LRU
        self._od.move_to_end(key, last=True)
        return item

    def put(self, key: str, item):
        # assign id
        if key not in self._key2id:
            self._key2id[key] = self._next_id
            self._id2key[self._next_id] = key
            self._next_id += 1

        # LRU insert/update
        existed = key in self._od
        self._od[key] = item
        self._od.move_to_end(key, last=True)

        # pooled vec
        self._vecs[key] = _pool_vec(item.e_cache)

        # evict
        evicted = None
        if len(self._od) > self.max_items:
            evicted, _ = self._od.popitem(last=False)
            self._vecs.pop(evicted, None)
            # id 映射保留也可以；但为了干净，删掉
            old_id = self._key2id.pop(evicted, None)
            if old_id is not None:
                self._id2key.pop(old_id, None)

        # 简化：有更新或驱逐就重建（max_items 小就很快）
        if existed or evicted is not None:
            self._rebuild_index()
        else:
            # 只 add 新向量：不重建
            idx = self._key2id[key]
            v = self._vecs[key].reshape(1, -1).astype(np.float32)
            self._index.add_with_ids(v, np.asarray([idx], dtype=np.int64))

    def search_best(self, tgt_e_cache, device, exact_topn: int = 8):
        """
        返回：(ref_key, ref_item, sim, debug)
        sim 用 pooled_cosine_01（0~1），hit 判断用 sim >= threshold（server 决定）
        """
        if len(self._od) == 0:
            return None, None, None, None

        q = _pool_vec(tgt_e_cache).reshape(1, -1).astype(np.float32)

        # 1) LSH 召回 topk
        k = min(self.topk, len(self._od))
        D, I = self._index.search(q, k)
        approx_ids = [int(x) for x in I[0] if int(x) != -1]
        approx_dist = [float(x) for x in D[0][: len(approx_ids)]]

        candidates: List[Tuple[str, object]] = []
        for idx in approx_ids:
            key = self._id2key.get(idx)
            if key is None:
                continue
            item = self._od.get(key)
            if item is None:
                continue
            candidates.append((key, item))

        # fallback：极端情况召回为空就退化到扫描
        if len(candidates) == 0:
            candidates = list(self._od.items())

        # 2) 精确重排：pooled_cosine_01 用的是 token-embedding 序列的 pooled cosine（你现有实现）
        # 为了复用你现有的 pooled_cosine_01（它接收 [L,D]），这里直接对 e_cache 做精确算
        best_key, best_item, best_sim = None, None, -1.0
        for k0, it in candidates[:exact_topn]:
            sim = pooled_cosine_01(tgt_e_cache, it.e_cache.to(device))
            if sim > best_sim:
                best_sim = float(sim)
                best_key, best_item = k0, it

        debug = LSHSearchDebug(approx_ids=approx_ids, approx_distances=approx_dist)
        return best_key, best_item, best_sim, debug
