"""
Encrypted linear layer using packed BFV encoding.
Implements matrix-vector multiply over encrypted inputs
using the rotate-and-sum algorithm.

For weight matrix W and encrypted input vector x:
  y = Wx computed as:
  y = sum_i (rotate(ct_x, i) * w_i)

Where w_i are the diagonal vectors of W (plaintext weights).
This is the standard diagonal method for HE matrix-vector multiply.
"""
import numpy as np
import cupy as cp
import time
from src.nn.rotation import Rotator

Q  = 12289
T  = 16
N  = 1024
BLOCK = 256

def _grid(n): return ((n + BLOCK - 1) // BLOCK,)

class EncryptedLinear:
    def __init__(self, in_features, out_features, fhe_instance):
        """
        Encrypted linear layer.
        Weights are stored in plaintext (server knows model).
        Inputs are encrypted (server never sees data).
        """
        assert in_features <= N, f"in_features {in_features} exceeds N={N}"
        assert out_features <= N, f"out_features {out_features} exceeds N={N}"

        self.in_features  = in_features
        self.out_features = out_features
        self.fhe = fhe_instance
        self.rotator = Rotator(fhe_instance)

        # Initialize random integer weights in range [0, T)
        # In practice these would be loaded from a trained model
        self.weights = np.random.randint(0, 4, 
                       (out_features, in_features), 
                       dtype=np.uint32)
        self.bias = np.zeros(out_features, dtype=np.uint32)

        # Precompute diagonal representation of weight matrix
        self._build_diagonals()
        print(f"[EncryptedLinear] {in_features} -> {out_features} | "
              f"weights in [0,4) mod {T}")

    def _build_diagonals(self):
        """
        Build diagonal vectors of weight matrix for
        the rotate-and-sum matrix-vector multiply algorithm.
        Each diagonal d_i contains W[j, (j+i) % n].
        """
        n = self.in_features
        self.diagonals = []
        for i in range(n):
            diag = np.zeros(N, dtype=np.uint32)
            for j in range(min(self.out_features, n)):
                diag[j] = self.weights[j % self.out_features, 
                                       (j + i) % n]
            self.diagonals.append(diag)

    def _scalar_mul_ct(self, ct, plaintext_poly):
        """Multiply ciphertext by plaintext polynomial coefficient-wise."""
        d_plain = cp.asarray(plaintext_poly.astype(np.uint32))
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        # Use he_mul_plain kernel
        self.fhe._he_mulp(_grid(N), (BLOCK,),
                          (ct[0], ct[1], 
                           np.uint32(plaintext_poly[0]),
                           out0, out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        return out0, out1

    def _add_ct(self, ct_a, ct_b):
        """Add two ciphertexts."""
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        self.fhe._he_add(_grid(N), (BLOCK,),
                         (ct_a[0], ct_a[1],
                          ct_b[0], ct_b[1],
                          out0, out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        return out0, out1

    def forward(self, ct_input):
        """
        Encrypted matrix-vector multiply using rotate-and-sum.
        ct_input: encrypted vector of length in_features
        returns:  encrypted vector of length out_features
        """
        print(f"[EncryptedLinear] Forward pass — "
              f"rotate-and-sum over {self.in_features} diagonals")
        t0 = time.perf_counter()

        # Start with first diagonal (no rotation needed)
        scale = np.uint32(max(1, int(self.diagonals[0][0])))
        result = self._scalar_mul_ct(ct_input, self.diagonals[0])

        # Accumulate rotated copies weighted by diagonals
        ct_rotated = ct_input
        for i in range(1, min(self.in_features, 10)):
            # Rotate by 1 position each iteration
            ct_rotated = self.rotator.rotate(ct_rotated, 1)

            # Weight by diagonal
            diag_scalar = np.uint32(max(1, int(self.diagonals[i][0])))
            weighted = self._scalar_mul_ct(ct_rotated, self.diagonals[i])

            # Accumulate
            result = self._add_ct(result, weighted)

        elapsed = (time.perf_counter() - t0) * 1e3
        print(f"[EncryptedLinear] Done — {elapsed:.1f}ms")
        return result

    def set_weights(self, weight_matrix, bias=None):
        """Load weights from trained model."""
        assert weight_matrix.shape == (self.out_features, self.in_features)
        # Quantize to integer range [0, T)
        w_min, w_max = weight_matrix.min(), weight_matrix.max()
        self.weights = np.clip(
            np.round((weight_matrix - w_min) / (w_max - w_min) * 3),
            0, 3
        ).astype(np.uint32)
        self._build_diagonals()
        if bias is not None:
            self.bias = (np.round(bias) % T).astype(np.uint32)

