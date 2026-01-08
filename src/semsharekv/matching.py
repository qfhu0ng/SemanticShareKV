import numpy as np
import torch
from .rope import build_rope_cache, apply_rope_to_embeddings
from .lsh import FaissLSHIndex

def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def prepare_fuzzy_mapping(
    ref_e: torch.Tensor,
    tgt_e: torch.Tensor,
    use_rope: bool = True,
    lsh_bits: int = 256,
    lsh_k: int = 8,
) -> dict:
    """
    ref_e: [Lr, D]
    tgt_e: [Lt, D]
    Return:
      {
        "map": LongTensor [Lt],
        "sim": FloatTensor [Lt]  (cosine sim of chosen match)
      }
    """
    device = tgt_e.device
    Lt, D = tgt_e.shape
    Lr = ref_e.shape[0]
    if ref_e.shape[1] != D:
        raise ValueError("ref/tgt embedding dim mismatch")

    if use_rope:
        cos, sin = build_rope_cache(max(Lr, Lt), D, device=device, dtype=tgt_e.dtype)
        ref_e2 = apply_rope_to_embeddings(ref_e, cos, sin)
        tgt_e2 = apply_rope_to_embeddings(tgt_e, cos, sin)
    else:
        ref_e2, tgt_e2 = ref_e, tgt_e

    ref_n = l2_normalize(ref_e2).float().cpu().numpy().astype(np.float32)
    tgt_n = l2_normalize(tgt_e2).float().cpu().numpy().astype(np.float32)

    index = FaissLSHIndex(dim=D, nbits=lsh_bits)
    index.build(ref_n)
    _, I = index.search(tgt_n, k=min(lsh_k, Lr))  # [Lt, k]

    ref_t = torch.from_numpy(ref_n).to(device=device)
    tgt_t = torch.from_numpy(tgt_n).to(device=device)
    I_t = torch.from_numpy(I).to(device=device)

    best = []
    sims = []
    for i in range(Lt):
        cand = I_t[i]
        cand_vec = ref_t[cand]
        s = (cand_vec @ tgt_t[i])  # [k]
        j = torch.argmax(s)
        best_idx = cand[j]
        best.append(best_idx)
        sims.append(s[j])

    return {"map": torch.stack(best).long(), "sim": torch.stack(sims).float()}

def rearrange_past_kv(past_key_values, mapping: torch.Tensor):
    """
    past_key_values: tuple(layers) of (k, v)
      k/v: [bs, n_kv_heads, Lr, head_dim]
    mapping: [Lt] target token -> reference token index
    returns: tuple(layers) of (k2, v2) where seq_len = Lt
    """
    new = []
    idx = mapping.view(1, 1, -1, 1)
    for (k, v) in past_key_values:
        k2 = k.gather(dim=2, index=idx.expand(k.size(0), k.size(1), -1, k.size(3))).contiguous()
        v2 = v.gather(dim=2, index=idx.expand(v.size(0), v.size(1), -1, v.size(3))).contiguous()
        new.append((k2, v2))
    return tuple(new)
