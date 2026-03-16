"""
Ciphertext rotation for packed BFV encoding.
Rotation by r slots uses the Galois automorphism x -> x^(5^r mod 2N)
combined with automorphism keys (similar to relin keys).
"""
import numpy as np
import cupy as cp

Q = 12289
N = 1024
BLOCK = 256

def _grid(n): return ((n + BLOCK - 1) // BLOCK,)

class Rotator:
    def __init__(self, fhe_instance):
        self.fhe = fhe_instance
        self._galois = None
        self._load_kernel()
        self._precompute_galois_keys()

    def _load_kernel(self):
        from pathlib import Path
        from src.gpu_utils import get_ptx
        kernels_dir = Path(__file__).parent.parent.parent / "kernels"
        ptx = get_ptx(kernels_dir, "fhe_kernel")
        mod = cp.RawModule(path=str(ptx))
        self._galois = mod.get_function("_Z19galois_automorphismPKjPjji")

    def _galois_element(self, r):
        """Compute 5^r mod 2N — the Galois element for rotation by r."""
        return pow(5, r, 2 * N)

    def _precompute_galois_keys(self):
        """
        Precompute automorphism keys for rotations.
        For each rotation amount we need keys similar to relin keys.
        We precompute for rotations 1..log2(N) which covers all
        needed rotations in the matrix-vector multiply.
        """
        self.galois_keys = {}
        sk = self.fhe.sk.astype(np.uint64)
        import math
        for r in range(1, int(math.log2(N)) + 1):
            k = self._galois_element(r)
            # Apply automorphism to secret key
            sk_rotated = np.zeros(N, dtype=np.uint64)
            for i in range(N):
                new_idx = (i * k) % (2 * N)
                if new_idx < N:
                    sk_rotated[new_idx] = sk[i]
                else:
                    sk_rotated[new_idx - N] = (Q - sk[i]) % Q
            a = np.random.randint(0, Q, N, dtype=np.uint64)
            e = np.random.randint(0, 3,  N, dtype=np.uint64)
            gk0 = (sk_rotated - a * sk % Q + e + 2*Q) % Q
            self.galois_keys[r] = (
                cp.asarray(gk0.astype(np.uint32)),
                cp.asarray(a.astype(np.uint32))
            )

    def rotate(self, ct, r):
        """Rotate ciphertext slots by r positions."""
        k = np.uint32(self._galois_element(r))
        ct0_rot = cp.zeros(N, dtype=cp.uint32)
        ct1_rot = cp.zeros(N, dtype=cp.uint32)
        self._galois(_grid(N), (BLOCK,),
                     (ct[0], ct0_rot, k, np.int32(N)))
        self._galois(_grid(N), (BLOCK,),
                     (ct[1], ct1_rot, k, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        return ct0_rot, ct1_rot
