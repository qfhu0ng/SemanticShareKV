import numpy as np

try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None


def _packbits_bool(b: np.ndarray) -> np.ndarray:
    """
    b: [N, nbits] bool
    return: [N, nbits/8] uint8
    """
    assert b.dtype == np.bool_
    assert b.ndim == 2
    return np.packbits(b, axis=1)


class FaissLSHIndex:
    """
    Backward-compatible LSH index expected by semsharekv.matching.py

    Implementation: Random hyperplane hashing (SimHash-style) -> FAISS binary index
    - codes are nbits binary, stored as packed bytes
    - search returns (I, D) where D is Hamming distance in bits
    """
    def __init__(self, dim: int, nbits: int = 256, seed: int = 1234):
        if faiss is None:
            raise RuntimeError("faiss is required (pip install faiss-cpu).")
        assert nbits % 8 == 0, "nbits must be multiple of 8"
        self.dim = int(dim)
        self.nbits = int(nbits)
        rng = np.random.RandomState(seed)
        self.R = rng.normal(size=(self.dim, self.nbits)).astype(np.float32)
        self.index = faiss.IndexBinaryFlat(self.nbits)
        self._built = False

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        x: [N, dim] float32
        -> [N, nbits/8] uint8 packed codes
        """
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        proj = x @ self.R  # [N, nbits]
        bits = proj > 0
        return _packbits_bool(bits)

    def build(self, ref_x: np.ndarray):
        """Reset + add reference vectors."""
        self.index.reset()
        codes = self.encode(ref_x)
        self.index.add(codes)
        self._built = True
        return codes

    def add(self, ref_x: np.ndarray):
        """Append reference vectors (without reset)."""
        codes = self.encode(ref_x)
        self.index.add(codes)
        self._built = True
        return codes

    def search(self, tgt_x: np.ndarray, k: int = 1):
        """
        Return:
          I: [Nt, k] int64
          D: [Nt, k] int32 (Hamming distance in bits)
        """
        if not self._built:
            raise RuntimeError("FaissLSHIndex not built yet. Call build() or add() first.")
        codes = self.encode(tgt_x)
        D, I = self.index.search(codes, int(k))
        return I, D

    @staticmethod
    def hamming_to_sim(D: np.ndarray, nbits: int) -> np.ndarray:
        """sim = 1 - D/nbits (clipped to [0,1])"""
        sim = 1.0 - (D.astype(np.float32) / float(nbits))
        return np.clip(sim, 0.0, 1.0)


def lsh_token_match_and_sim(ref_e: np.ndarray, tgt_e: np.ndarray, nbits: int = 256, seed: int = 1234):
    """
    ref_e: [Lr, D]
    tgt_e: [Lt, D]
    Returns:
      mapping: [Lt] each tgt token -> nearest ref token index under Hamming
      sim: scalar in [0,1] = 1 - mean_hamming/nbits
    """
    if ref_e.dtype != np.float32:
        ref_e = ref_e.astype(np.float32)
    if tgt_e.dtype != np.float32:
        tgt_e = tgt_e.astype(np.float32)

    lsh = FaissLSHIndex(dim=ref_e.shape[1], nbits=nbits, seed=seed)
    lsh.build(ref_e)
    I, D = lsh.search(tgt_e, k=1)  # [Lt,1]
    mapping = I.reshape(-1).astype(np.int64)

    mean_hamming = float(D.mean())
    sim = 1.0 - (mean_hamming / float(nbits))
    sim = max(0.0, min(1.0, sim))
    return mapping, sim
