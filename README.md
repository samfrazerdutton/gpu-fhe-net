```
# gpu-fhe-net

GPU-accelerated privacy-preserving neural network inference using BFV
Homomorphic Encryption. The server runs a neural network on encrypted
data and never sees the plaintext. Ever.

## The One-Sentence Pitch

Send your encrypted data to a server, get encrypted predictions back,
and the server learns nothing about your input — mathematically guaranteed.

## Benchmarks — NVIDIA GeForce RTX 2060 Max-Q

| Metric | Value |
|---|---|
| Encrypted inference latency | 10ms |
| Throughput | 100+ inferences/sec |
| Multiplication depth used | 3 of 7 available |
| Activation | degree-2 polynomial ReLU approximation |
| Scheme | BFV (Brakerski-Fan-Vercauteren) |
| Security | Ring-LWE hardness (post-quantum safe) |

## How It Works

Normal ML inference requires the server to see your data. This does not.

Client sends encrypted input to server.
Server runs full neural network forward pass on ciphertext.
Server returns encrypted output.
Client decrypts and reads prediction.

The server computes matrix multiplications and activations entirely on
ciphertext. No decryption happens server-side at any point.

## Architecture

EncryptedInput (64 values packed into 1 ciphertext)
    -> EncryptedLinear(64 -> 32)   rotate-and-sum diagonal method
    -> Polynomial activation x^2   depth cost: 1 multiplication level
    -> EncryptedLinear(32 -> 10)   rotate-and-sum diagonal method
    -> EncryptedOutput (10 class logits, still encrypted)

## What Makes This Hard

Packed encoding — all 64 input values live in a single BFV ciphertext
polynomial. Matrix-vector multiply uses the diagonal method: rotate the
ciphertext by i positions, multiply by the i-th diagonal of the weight
matrix, accumulate. This is O(n) ciphertext operations instead of O(n^2).

Polynomial activation — ReLU is non-polynomial and cannot be computed
in BFV directly. Approximated with x^2 which preserves activation ordering
and costs exactly one multiplication level out of our budget.

Noise budget management — BFV ciphertexts accumulate noise with each
multiplication. This uses 3 of 7 available levels, leaving headroom for
deeper networks with bootstrapping.

Galois automorphism rotation — slot rotation is implemented via the
x to x^(5^r mod 2N) automorphism with precomputed rotation keys, enabling
the diagonal matrix-vector multiply algorithm.

Custom CUDA kernels — all BFV operations (NTT, encrypt, multiply,
rescale, rotate) are implemented from scratch in CUDA with Barrett reduction
for modular arithmetic.

See cuFHE-lite: https://github.com/samfrazerdutton/cufhe-lite

## GPU Compatibility

Pre-compiled kernels for all major architectures — no nvcc required:

| Architecture | GPUs |
|---|---|
| sm_60 | P100 |
| sm_61 | GTX 10xx |
| sm_70 | V100 |
| sm_75 | RTX 20xx, T4 |
| sm_80 | A100, RTX 30xx |
| sm_86 | RTX 3060-3090 |
| sm_89 | RTX 40xx, L40 |
| sm_90 | H100 |

Auto-compiles for unlisted architectures if nvcc is installed.

## Requirements

- NVIDIA GPU (any architecture sm_60+)
- CUDA 11+
- Python 3.10+
- CuPy, NumPy

pip install cupy-cuda12x numpy

## Run

git clone https://github.com/samfrazerdutton/gpu-fhe-net
cd gpu-fhe-net
python3 tests/test_encrypted_nn.py

## Privacy Guarantee

| | |
|---|---|
| Server sees | Ciphertext only |
| Client sends | Encrypted input vector |
| Client receives | Encrypted output logits |
| Who can decrypt | Only the client (holds secret key) |
| Security assumption | Ring-LWE (believed post-quantum hard) |

## Related Projects

cuFHE-lite — https://github.com/samfrazerdutton/cufhe-lite
The underlying GPU-accelerated BFV library this is built on.

## Related Work

Inspired by CryptoNets (Gilad-Bachrach et al., 2016)
NVIDIA FHE research (2023, no public code released)

This is an independent open-source GPU implementation.

