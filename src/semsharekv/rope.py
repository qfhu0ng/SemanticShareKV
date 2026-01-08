import torch

def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0, device=None, dtype=None):
    """
    Build cos/sin for "rotary-like" positional injection into embeddings (for matching).
    dim must be even.
    Returns: cos, sin with shape [seq_len, dim/2]
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    device = device or "cpu"
    dtype = dtype or torch.float32

    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [seq_len, half]
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope_to_embeddings(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply RoPE-style rotation to embeddings x.
    x: [seq_len, dim] or [bs, seq_len, dim]
    cos/sin: [seq_len, dim/2]
    """
    squeeze_back = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_back = True

    bs, seq_len, dim = x.shape
    if dim % 2 != 0:
        raise ValueError("Embedding dim must be even for RoPE.")
    half = dim // 2

    cos = cos[:seq_len].unsqueeze(0)  # [1, seq_len, half]
    sin = sin[:seq_len].unsqueeze(0)

    x1 = x[..., :half]
    x2 = x[..., half:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.cat([y1, y2], dim=-1)

    return y.squeeze(0) if squeeze_back else y
