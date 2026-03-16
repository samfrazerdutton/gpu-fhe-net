"""
Encrypted Neural Network for MNIST inference using BFV FHE.
Architecture: 784 -> 128 -> 10
Activation: x^2 polynomial approximation of ReLU
All computation happens on encrypted data — server never
sees plaintext inputs or outputs.
"""
import numpy as np
import cupy as cp
import time
from src.fhe_bridge import cuFHE
from src.nn.layers import EncryptedLinear
from src.nn.chebyshev import approx_relu_bfv

T = 16
N = 1024

class EncryptedMNISTNet:
    def __init__(self, fhe_instance=None):
        print("\n[EncryptedNet] Initializing encrypted MNIST network...")
        self.fhe = fhe_instance if fhe_instance else cuFHE()

        # Two layer network — fits within depth budget
        # 784->128 and 128->10
        # We use smaller dimensions for feasibility with Q=12289
        self.layer1 = EncryptedLinear(64, 32, self.fhe)
        self.layer2 = EncryptedLinear(32, 10, self.fhe)

        print("[EncryptedNet] Ready — 64 -> 32 -> 10")
        print("[EncryptedNet] Activation: x^2 polynomial ReLU approx")
        print("[EncryptedNet] Depth budget: layer1=1, act=1, layer2=1 = 3 total")

    def encrypt_input(self, x: np.ndarray) -> tuple:
        """
        Encrypt input vector.
        Quantizes float inputs to [0, T) integer range.
        Packs into single ciphertext using polynomial slots.
        """
        assert len(x) <= N, f"Input length {len(x)} exceeds N={N}"
        # Quantize to integer range
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x_int = np.round(
                (x - x_min) / (x_max - x_min) * (T - 1)
            ).astype(np.uint32)
        else:
            x_int = np.zeros_like(x, dtype=np.uint32)

        # Pad to N
        padded = np.zeros(N, dtype=np.uint32)
        padded[:len(x)] = x_int
        return self.fhe.encrypt(padded)

    def forward(self, ct_input) -> tuple:
        """
        Encrypted forward pass.
        Input and all intermediate values stay encrypted.
        """
        print("\n[EncryptedNet] === Encrypted Forward Pass ===")
        t_total = time.perf_counter()

        # Layer 1
        print("[EncryptedNet] Layer 1: encrypted linear...")
        ct = self.layer1.forward(ct_input)

        # Activation — x^2 approximation
        print("[EncryptedNet] Activation: x^2 (polynomial ReLU approx)...")
        t0 = time.perf_counter()
        ct = approx_relu_bfv(ct, self.fhe)
        print(f"[EncryptedNet] Activation done — "
              f"{(time.perf_counter()-t0)*1e3:.1f}ms")

        # Layer 2
        print("[EncryptedNet] Layer 2: encrypted linear...")
        ct = self.layer2.forward(ct)

        elapsed = (time.perf_counter() - t_total) * 1e3
        print(f"\n[EncryptedNet] Total forward pass: {elapsed:.1f}ms")
        return ct

    def decrypt_output(self, ct) -> np.ndarray:
        """Decrypt and return class logits."""
        out = self.fhe.decrypt(*ct)
        return out[:10]

    def predict(self, x: np.ndarray) -> int:
        """Full pipeline: encrypt -> forward -> decrypt -> argmax."""
        ct_in  = self.encrypt_input(x)
        ct_out = self.forward(ct_in)
        logits = self.decrypt_output(ct_out)
        return int(np.argmax(logits)), logits

    def benchmark_inference(self, n_samples=10):
        """Benchmark encrypted inference time."""
        print(f"\n[EncryptedNet] Benchmarking {n_samples} encrypted inferences...")
        x = np.random.rand(64).astype(np.float32)
        times = []
        for i in range(n_samples):
            t0 = time.perf_counter()
            pred, logits = self.predict(x)
            elapsed = (time.perf_counter() - t0) * 1e3
            times.append(elapsed)
            print(f"  Sample {i+1}: pred={pred} | {elapsed:.1f}ms")
        print(f"\n[EncryptedNet] Mean inference time: {np.mean(times):.1f}ms")
        print(f"[EncryptedNet] Throughput: {1000/np.mean(times):.2f} inferences/sec")

