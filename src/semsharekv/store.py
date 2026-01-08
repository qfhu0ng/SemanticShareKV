from dataclasses import dataclass
from collections import OrderedDict
import torch

@dataclass
class CacheItem:
    prompt: str
    e_cache: torch.Tensor   # [L, D] on CPU
    past_kv: tuple          # tuple(layers)->(k,v), on CPU

class LRUCacheStore:
    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self._store = OrderedDict()

    def put(self, key: str, item: CacheItem):
        if key in self._store:
            self._store.pop(key)
        self._store[key] = item
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def items(self):
        return list(self._store.items())

    def __len__(self):
        return len(self._store)

@torch.no_grad()
def pooled_cosine_01(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """
    Approx similarity in [0,1] using mean-pooled cosine.
    """
    v1 = e1.mean(dim=0)
    v2 = e2.mean(dim=0)
    sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()
    return 0.5 * (sim + 1.0)
