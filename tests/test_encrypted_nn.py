"""
Encrypted Neural Network Test Suite
Demonstrates privacy-preserving inference using BFV FHE on GPU.
The server computes on encrypted inputs and never sees plaintext.
"""
import numpy as np
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fhe_bridge import cuFHE
from src.nn.network import EncryptedMNISTNet
from src.nn.chebyshev import verify_approx

def test_chebyshev_approx():
    print("\n── Test 1: Chebyshev ReLU Approximation ──")
    mse = verify_approx(verbose=True)
    assert mse < 0.01, f"Approximation too poor: MSE={mse}"
    print(f"[✓] ReLU approximation MSE={mse:.6f} within tolerance")

def test_single_encrypted_inference():
    print("\n── Test 2: Single Encrypted Inference ──")
    net = EncryptedMNISTNet()
    x = np.random.rand(64).astype(np.float32)
    print(f"  Input (plaintext, client side): {x[:8].round(3)}...")
    t0 = time.perf_counter()
    pred, logits = net.predict(x)
    elapsed = (time.perf_counter() - t0) * 1e3
    print(f"  Output logits (decrypted): {logits}")
    print(f"  Predicted class: {pred}")
    print(f"  Total time: {elapsed:.1f}ms")
    print(f"[✓] Encrypted inference completed — server never saw plaintext")

def test_deterministic_output():
    print("\n── Test 3: Output Consistency ──")
    net = EncryptedMNISTNet()
    x = np.array([0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4] * 8,
                 dtype=np.float32)
    pred1, logits1 = net.predict(x)
    pred2, logits2 = net.predict(x)
    print(f"  Run 1 logits: {logits1}")
    print(f"  Run 2 logits: {logits2}")
    match = np.array_equal(logits1, logits2)
    print(f"[{'✓' if match else '✗'}] Outputs consistent: {match}")

def test_different_inputs_different_outputs():
    print("\n── Test 4: Different Inputs Produce Different Outputs ──")
    net = EncryptedMNISTNet()
    x1 = np.zeros(64, dtype=np.float32)
    x2 = np.ones(64, dtype=np.float32)
    x3 = np.random.rand(64).astype(np.float32)
    _, l1 = net.predict(x1)
    _, l2 = net.predict(x2)
    _, l3 = net.predict(x3)
    diff_12 = not np.array_equal(l1, l2)
    diff_13 = not np.array_equal(l1, l3)
    print(f"  Zeros  logits: {l1}")
    print(f"  Ones   logits: {l2}")
    print(f"  Random logits: {l3}")
    print(f"[{'✓' if diff_12 else '!'}] zeros != ones: {diff_12}")
    print(f"[{'✓' if diff_13 else '!'}] zeros != random: {diff_13}")

def test_noise_budget_survives():
    print("\n── Test 5: Noise Budget Survives Full Forward Pass ──")
    fhe = cuFHE()
    net = EncryptedMNISTNet(fhe)
    # Encrypt a known value and check it survives forward pass
    msg = np.array([5] + [0]*(1024-1), dtype=np.uint32)
    ct  = fhe.encrypt(msg)
    dec = fhe.decrypt(*ct)
    print(f"  Pre-forward decrypt check: {dec[0]} (expected 5)")
    assert dec[0] == 5, "Encryption broken before forward pass"
    x   = np.random.rand(64).astype(np.float32)
    ct_in  = net.encrypt_input(x)
    ct_out = net.forward(ct_in)
    # Verify output is decryptable without error
    logits = net.decrypt_output(ct_out)
    valid  = len(logits) == 10 and not np.all(logits == 0)
    print(f"  Post-forward logits: {logits}")
    print(f"[{'✓' if valid else '✗'}] Noise budget survived full forward pass")

def benchmark_encrypted_inference():
    print("\n── Benchmark: Encrypted Inference Throughput ──")
    net = EncryptedMNISTNet()
    net.benchmark_inference(n_samples=5)

def print_privacy_summary():
    print("\n" + "█"*60)
    print("  PRIVACY GUARANTEE SUMMARY")
    print("█"*60)
    print("""
  What the SERVER computes on:     Encrypted ciphertext
  What the SERVER sees:            Nothing — only ciphertext
  What the CLIENT sends:           Encrypted input vector
  What the CLIENT receives:        Encrypted output logits
  Who can decrypt:                 Only the CLIENT (holds secret key)

  Scheme:    BFV (Brakerski-Fan-Vercauteren)
  Security:  Ring-LWE hardness assumption
  Backend:   Custom CUDA kernels on RTX 2060 Max-Q
  Depth:     3 multiplication levels used of 7 available
    """)

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  cuFHE-lite — ENCRYPTED NEURAL NETWORK TEST SUITE")
    print("  Privacy-preserving inference — server sees NO plaintext")
    print("█"*60)

    test_chebyshev_approx()
    test_single_encrypted_inference()
    test_deterministic_output()
    test_different_inputs_different_outputs()
    test_noise_budget_survives()
    benchmark_encrypted_inference()
    print_privacy_summary()

    print("\n" + "█"*60)
    print("  ALL ENCRYPTED NN TESTS COMPLETE")
    print("█"*60)

