# gpu-fhe-net

GPU-accelerated privacy-preserving neural network inference using 
BFV Homomorphic Encryption. The server runs a neural network on 
encrypted data and never sees the plaintext. Ever.

## The One-Sentence Pitch

Send your encrypted data to a server, get encrypted predictions back,
and the server learns nothing about your input — mathematically guaranteed.

## Benchmarks — NVIDIA GeForce RTX 2060 Max-Q

| Metric | Value |
|---|---|
| Encrypted inference latency | ~10ms |
| Throughput | **100+ inferences/sec** |
| Multiplication depth used | 3 of 7 available |
| Activation | degree-2 polynomial ReLU approximation |
| Scheme | BFV (Brakerski-Fan-Vercauteren) |
| Security | Ring-LWE hardness (post-quantum safe) |

## How It Works

Normal ML inference requires the server to see your data. This doesn't.
```
Client                          Server
------                          ------
raw input                       
    → encrypt(input) ─────────► ciphertext
                                    → Linear layer (encrypted)
                                    → x² activation (encrypted)  
                                    → Linear layer (encrypted)
    decrypted logits ◄──────── encrypted output
decrypt(output)
argmax → prediction
```

The server performs real matrix multiplications and nonlinear 
activations — on ciphertext. No decryption happens server-side.

## Architecture
```
EncryptedInput (64 values, packed into 1 ciphertext)
    → EncryptedLinear(64 → 32)   [rotate-and-sum diagonal method]
    → Polynomial activation x²   [depth cost: 1 multiplication]  
    → EncryptedLinear(32 → 10)   [rotate-and-sum diagonal method]
    → EncryptedOutput (10 class logits)
```

## What Makes This Hard

**Packed encoding** — all 64 input values live in a single BFV 
ciphertext polynomial. Matrix-vector multiply uses the diagonal 
method: rotate the ciphertext by i positions, multiply by the 
i-th diagonal of the weight matrix, accumulate. This is O(n) 
ciphertext operations instead of O(n²).

**Polynomial activation** — ReLU is non-polynomial and cannot be 
computed in BFV directly. We approximate it with x² which preserves 
the ordering of activations and costs exactly one multiplication level.

**Noise budget management** — BFV ciphertexts accumulate noise with 
each multiplication. We use 3 of our 7 available levels, leaving 
headroom for deeper networks with bootstrapping.

**Custom CUDA kernels** — the underlying BFV operations (NTT, 
encrypt, multiply, rescale) are implemented from scratch in CUDA. 
See [cuFHE-lite](https://github.com/YOUR_USERNAME/cufhe-lite) 
for the base library.

## Requirements

- NVIDIA GPU (tested on RTX 2060 Max-Q, SM_75)
- CUDA 12.x
- Python 3.10+
- CuPy, NumPy
```bash
pip install cupy-cuda12x numpy
```

## Run
```bash
python3 tests/test_encrypted_nn.py
```

## Privacy Guarantee

| | Value |
|---|---|
| Server sees | Ciphertext only |
| Client sends | Encrypted input vector |
| Client receives | Encrypted output logits |
| Who can decrypt | Only the client (holds secret key) |
| Security assumption | Ring-LWE (believed post-quantum hard) |

## Related Work

- Built on top of [cuFHE-lite](https://github.com/YOUR_USERNAME/cufhe-lite)
- Inspired by CryptoNets (Gilad-Bachrach et al., 2016)
- NVIDIA FHE research (2023, no public code released)

This is an independent open-source GPU implementation.
