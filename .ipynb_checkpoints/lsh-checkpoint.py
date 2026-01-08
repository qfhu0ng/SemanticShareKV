import numpy as np
import faiss

class FaissLSHIndex:
    """
    Simple FAISS LSH-based ANN for token vectors.
    """
    def __init__(self, dim: int, nbits: int = 256):
        self.dim = dim
        self.nbits = nbits
        self.index = faiss.IndexLSH(dim, nbits)
        self._built = False

    def build(self, xb: np.ndarray):
        """
        xb: [N, dim] float32
        """
        assert xb.dtype == np.float32 and xb.ndim == 2 and xb.shape[1] == self.dim
        self.index.add(xb)
        self._built = True

    def search(self, xq: np.ndarray, k: int = 8):
        """
        xq: [M, dim] float32
        Returns: (D, I) with shapes [M, k]
        """
        assert self._built, "Call build() first."
        return self.index.search(xq, k)
