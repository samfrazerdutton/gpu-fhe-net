#include <stdint.h>
#include <cuda_runtime.h>

#define Q      12289ULL
#define T      16ULL
#define N      1024
#define DELTA  768ULL    // floor(Q/T) = floor(12289/16) = 768

__device__ __forceinline__ uint32_t reduce_q(uint64_t a) {
    return (uint32_t)(a % Q);
}

__global__ void poly_add(const uint32_t* a, const uint32_t* b,
                         uint32_t* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = (uint32_t)((a[i] + b[i]) % Q);
}

__global__ void poly_sub(const uint32_t* a, const uint32_t* b,
                         uint32_t* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = (uint32_t)((a[i] + Q - b[i]) % Q);
}

__global__ void poly_scalar_mul(const uint32_t* a, uint32_t s,
                                uint32_t* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = (uint32_t)((uint64_t)a[i] * s % Q);
}

__global__ void bfv_encrypt(const uint32_t* msg, const uint32_t* err,
                             uint32_t* ct0, uint32_t* ct1, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ct0[i] = (uint32_t)((msg[i] * DELTA + err[i]) % Q);
    ct1[i] = 0;
}

__global__ void bfv_decrypt(const uint32_t* ct0, uint32_t* msg, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t v = (uint64_t)ct0[i] * T + Q / 2;
    msg[i] = (uint32_t)((v / Q) % T);
}

__global__ void he_add(const uint32_t* a0, const uint32_t* a1,
                       const uint32_t* b0, const uint32_t* b1,
                       uint32_t* c0, uint32_t* c1, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c0[i] = (uint32_t)((a0[i] + b0[i]) % Q);
    c1[i] = (uint32_t)((a1[i] + b1[i]) % Q);
}

__global__ void he_mul_plain(const uint32_t* ct0, const uint32_t* ct1,
                             uint32_t scalar,
                             uint32_t* out0, uint32_t* out1, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out0[i] = (uint32_t)((uint64_t)ct0[i] * scalar % Q);
    out1[i] = (uint32_t)((uint64_t)ct1[i] * scalar % Q);
}

// Negacyclic NTT forward
// pre_mult: psi^i twiddle factors (length N)
// roots:    omega^bitrev(i) twiddle table (length N)
__global__ void ntt_forward(uint32_t* poly,
                            const uint32_t* pre_mult,
                            const uint32_t* roots,
                            int n, int stage) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int half   = 1 << stage;
    int stride = half * 2;
    int group  = tid / half;
    int pos    = tid % half;
    int i = group * stride + pos;
    int j = i + half;
    if (j >= n) return;
    uint32_t w = roots[half + pos];
    uint64_t u = poly[i];
    uint64_t v = (uint64_t)poly[j] * w % Q;
    poly[i] = (uint32_t)((u + v) % Q);
    poly[j] = (uint32_t)((u + Q - v) % Q);
}

// Pre-multiply by psi^i before NTT (negacyclic twist)
__global__ void ntt_premul(uint32_t* poly, const uint32_t* psi_pow, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    poly[i] = (uint32_t)((uint64_t)poly[i] * psi_pow[i] % Q);
}

// Post-multiply by inv_psi^i after INTT (negacyclic untwist)
__global__ void ntt_postmul(uint32_t* poly, const uint32_t* inv_psi_pow,
                            uint32_t inv_n, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t v = (uint64_t)poly[i] * inv_n % Q;
    poly[i] = (uint32_t)(v * inv_psi_pow[i] % Q);
}

__global__ void ntt_inverse(uint32_t* poly,
                            const uint32_t* inv_roots,
                            int n, int stage) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int half   = 1 << stage;
    int stride = half * 2;
    int group  = tid / half;
    int pos    = tid % half;
    int i = group * stride + pos;
    int j = i + half;
    if (j >= n) return;
    uint32_t w = inv_roots[half + pos];
    uint64_t u = poly[i];
    uint64_t v = poly[j];
    poly[i] = (uint32_t)((u + v) % Q);
    poly[j] = (uint32_t)((u + Q - v) % Q * w % Q);
}

__global__ void poly_pointwise_mul(const uint32_t* a, const uint32_t* b,
                                   uint32_t* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = (uint32_t)((uint64_t)a[i] * b[i] % Q);
}

__global__ void poly_scale(uint32_t* poly, uint32_t inv_n, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    poly[i] = (uint32_t)((uint64_t)poly[i] * inv_n % Q);
}

// BFV rescaling: round(T/Q * x) stays in Z_Q
// DELTA_INV = DELTA^{-1} mod Q = 768^{-1} mod 12289 = 12273
// Rescaling: multiply by DELTA_INV to go from DELTA^2*m -> DELTA*m
#define DELTA_INV 12273ULL

__global__ void bfv_rescale(const uint32_t* input, uint32_t* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = (uint32_t)((uint64_t)input[i] * DELTA_INV % Q);
}

__global__ void relin_key_mul(const uint32_t* ct2,
                              const uint32_t* rlk0, const uint32_t* rlk1,
                              uint32_t* out0, uint32_t* out1, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out0[i] = (uint32_t)((uint64_t)ct2[i] * rlk0[i] % Q);
    out1[i] = (uint32_t)((uint64_t)ct2[i] * rlk1[i] % Q);
}

#define Q_PRIME 257U

__global__ void modswitch_down(const uint32_t* in, uint32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t scaled = (uint64_t)in[i] * Q_PRIME + Q / 2;
    out[i] = (uint32_t)((scaled / Q) % Q_PRIME);
}

__global__ void modswitch_up(const uint32_t* in, uint32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i] % (uint32_t)Q;
}
// Galois automorphism for ciphertext rotation
// Applies x -> x^k map to polynomial coefficients
// For rotation by r: k = 5^r mod 2N (generator of Galois group)
__global__ void galois_automorphism(const uint32_t* in, uint32_t* out,
                                     uint32_t k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t new_idx = ((uint64_t)i * k) % (2 * n);
    if (new_idx < (uint32_t)n) {
        out[new_idx] = in[i];
    } else {
        out[new_idx - n] = (Q - in[i]) % Q;
    }
}
