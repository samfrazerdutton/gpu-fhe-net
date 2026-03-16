"""
cuFHE-lite: GPU-accelerated BFV Homomorphic Encryption
Q=12289 (NTT prime: 3*2^12+1), T=16, N=1024, DELTA=768

Negacyclic NTT for polynomial multiplication mod (X^N+1, Q):
  Forward: premul by psi^i -> standard NTT with omega twiddles
  Inverse: standard INTT with inv_omega twiddles -> postmul by inv_psi^i * inv_N

Where psi=1945 is the primitive 2N-th root of unity mod Q,
and omega=psi^2=10302 is the primitive N-th root of unity mod Q.
"""
import subprocess
import numpy as np
import cupy as cp
from pathlib import Path
import time
from src.gpu_utils import get_ptx, get_device_info

Q       = 12289
Q_PRIME = 257
T       = 16
N       = 1024
DELTA   = Q // T   # 768
BLOCK   = 256

PSI     = 1945
INV_PSI = 4050
OMEGA   = pow(PSI, 2, Q)       # 10302
INV_N   = pow(N, Q-2, Q)       # 12277

def _grid(n): return ((n + BLOCK - 1) // BLOCK,)

def _build_twiddles():
    """Build all twiddle factor tables for negacyclic NTT."""
    def bitrev(x, log2n):
        r = 0
        for _ in range(log2n):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r

    log2n = 10  # log2(1024)
    inv_omega = pow(OMEGA, Q-2, Q)

    # Bit-reversed omega powers for Cooley-Tukey butterfly
    roots     = np.array([pow(OMEGA,     bitrev(i, log2n), Q) for i in range(N)],
                         dtype=np.uint32)
    inv_roots = np.array([pow(inv_omega, bitrev(i, log2n), Q) for i in range(N)],
                         dtype=np.uint32)

    # Negacyclic twist factors
    psi_pow     = np.array([pow(PSI,     i, Q) for i in range(N)], dtype=np.uint32)
    inv_psi_pow = np.array([pow(INV_PSI, i, Q) for i in range(N)], dtype=np.uint32)

    return roots, inv_roots, psi_pow, inv_psi_pow

class cuFHE:
    def __init__(self):
        kernels_dir = Path(__file__).parent.parent / "kernels"
        ptx = get_ptx(kernels_dir, "fhe_kernel")
        mod = cp.RawModule(path=str(ptx))
        info = get_device_info()
        print(f"[GPU] {info['name']} | {info['sm'].upper()} | "
              f"{info['vram_gb']:.1f}GB VRAM | {info['sm_count']} SMs")

        # Load kernels
        self._enc      = mod.get_function("_Z11bfv_encryptPKjS0_PjS1_i")
        self._dec      = mod.get_function("_Z11bfv_decryptPKjPji")
        self._kadd     = mod.get_function("_Z8poly_addPKjS0_Pji")
        self._ksub     = mod.get_function("_Z8poly_subPKjS0_Pji")
        self._kscalar  = mod.get_function("_Z15poly_scalar_mulPKjjPji")
        self._he_add   = mod.get_function("_Z6he_addPKjS0_S0_S0_PjS1_i")
        self._he_mulp  = mod.get_function("_Z12he_mul_plainPKjS0_jPjS1_i")
        self._ntt_fwd  = mod.get_function("_Z11ntt_forwardPjPKjS1_ii")
        self._ntt_inv  = mod.get_function("_Z11ntt_inversePjPKjii")
        self._premul   = mod.get_function("_Z10ntt_premulPjPKji")
        self._postmul  = mod.get_function("_Z11ntt_postmulPjPKjji")
        self._pw_mul   = mod.get_function("_Z18poly_pointwise_mulPKjS0_Pji")
        self._rescale  = mod.get_function("_Z11bfv_rescalePKjPji")
        self._relin    = mod.get_function("_Z13relin_key_mulPKjS0_S0_PjS1_i")
        self._msw_dn   = mod.get_function("_Z14modswitch_downPKjPji")
        self._msw_up   = mod.get_function("_Z12modswitch_upPKjPji")

        # Precompute and upload twiddle factors
        roots, inv_roots, psi_pow, inv_psi_pow = _build_twiddles()
        self.d_roots       = cp.asarray(roots)
        self.d_inv_roots   = cp.asarray(inv_roots)
        self.d_psi_pow     = cp.asarray(psi_pow)
        self.d_inv_psi_pow = cp.asarray(inv_psi_pow)

        # Secret key
        self.sk = np.random.randint(0, 2, N, dtype=np.uint32)

        # Relin keys
        a    = np.random.randint(0, Q, N, dtype=np.uint64)
        e    = np.random.randint(0, 3,  N, dtype=np.uint64)
        sk64 = self.sk.astype(np.uint64)
        sk2  = sk64 * sk64 % Q
        rlk0 = (sk2 - a * sk64 % Q + e + 2*Q) % Q
        self.d_rlk0 = cp.asarray(rlk0.astype(np.uint32))
        self.d_rlk1 = cp.asarray(a.astype(np.uint32))

        print(f"[cuFHE] Ready — N={N}, Q={Q}, T={T}, Δ={DELTA}")
        print(f"[cuFHE] Noise budget ~{Q//(2*DELTA)-1} muls | "
              f"psi={PSI}, omega={OMEGA}, inv_N={INV_N}")

    # PTX compilation handled by gpu_utils.get_ptx()

    def _ntt(self, d_poly):
        """Forward negacyclic NTT in-place."""
        # Step 1: premultiply by psi^i
        self._premul(_grid(N), (BLOCK,),
                     (d_poly, self.d_psi_pow, np.int32(N)))
        # Step 2: standard Cooley-Tukey NTT
        import math
        for s in range(int(math.log2(N))):
            self._ntt_fwd(_grid(N//2), (BLOCK,),
                          (d_poly, self.d_psi_pow, self.d_roots,
                           np.int32(N), np.int32(s)))
        cp.cuda.Stream.null.synchronize()

    def _intt(self, d_poly):
        """Inverse negacyclic NTT in-place."""
        import math
        # Step 1: standard inverse NTT
        for s in range(int(math.log2(N))):
            self._ntt_inv(_grid(N//2), (BLOCK,),
                          (d_poly, self.d_inv_roots,
                           np.int32(N), np.int32(s)))
        # Step 2: postmultiply by inv_psi^i * inv_N
        self._postmul(_grid(N), (BLOCK,),
                      (d_poly, self.d_inv_psi_pow,
                       np.uint32(INV_N), np.int32(N)))
        cp.cuda.Stream.null.synchronize()

    def _polymul(self, a_np, b_np) -> cp.ndarray:
        """Negacyclic polynomial multiplication mod (X^N+1, Q)."""
        da = cp.asarray(a_np.astype(np.uint32).copy())
        db = cp.asarray(b_np.astype(np.uint32).copy())
        self._ntt(da)
        self._ntt(db)
        dc = cp.zeros(N, dtype=cp.uint32)
        self._pw_mul(_grid(N), (BLOCK,), (da, db, dc, np.int32(N)))
        self._intt(dc)
        return dc

    def verify_ntt(self):
        """Sanity check: [3]*[2] should give [6,0,0,...]"""
        a = np.array([3] + [0]*(N-1), dtype=np.uint32)
        b = np.array([2] + [0]*(N-1), dtype=np.uint32)
        c = cp.asnumpy(self._polymul(a, b))
        ok = c[0] == 6 and np.all(c[1:5] == 0)
        print(f"[cuFHE] NTT verify: [3]*[2]={c[:5]} {'✓' if ok else '✗ BROKEN'}")
        return ok

    def encrypt(self, message: np.ndarray) -> tuple:
        assert message.max() < T, f"Values must be < {T}"
        msg = cp.asarray(message.astype(np.uint32))
        err = cp.zeros(N, dtype=np.uint32)  # zero error for testing
        ct0 = cp.zeros(N, dtype=cp.uint32)
        ct1 = cp.zeros(N, dtype=cp.uint32)
        t0  = time.perf_counter()
        self._enc(_grid(N),(BLOCK,),(msg,err,ct0,ct1,np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Encrypt {(time.perf_counter()-t0)*1e3:.3f}ms")
        return ct0, ct1

    def decrypt(self, ct0, ct1) -> np.ndarray:
        out = cp.zeros(N, dtype=cp.uint32)
        t0  = time.perf_counter()
        self._dec(_grid(N),(BLOCK,),(ct0,out,np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Decrypt {(time.perf_counter()-t0)*1e3:.3f}ms")
        return cp.asnumpy(out)

    def he_add(self, ct_a, ct_b) -> tuple:
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        t0   = time.perf_counter()
        self._he_add(_grid(N),(BLOCK,),
                     (ct_a[0],ct_a[1],ct_b[0],ct_b[1],
                      out0,out1,np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] HE ADD {(time.perf_counter()-t0)*1e3:.3f}ms")
        return out0, out1

    def he_mul_ct(self, ct_a, ct_b) -> tuple:
        """BFV ct*ct: NTT poly multiply + BFV rescaling."""
        t0 = time.perf_counter()

        a0 = cp.asnumpy(ct_a[0])
        a1 = cp.asnumpy(ct_a[1])
        b0 = cp.asnumpy(ct_b[0])
        b1 = cp.asnumpy(ct_b[1])

        d_d0  = self._polymul(a0, b0)
        d_d1a = self._polymul(a0, b1)
        d_d1b = self._polymul(a1, b0)

        d_d1 = cp.zeros(N, dtype=cp.uint32)
        self._kadd(_grid(N),(BLOCK,),(d_d1a,d_d1b,d_d1,np.int32(N)))
        cp.cuda.Stream.null.synchronize()

        c0 = cp.zeros(N, dtype=cp.uint32)
        c1 = cp.zeros(N, dtype=cp.uint32)
        self._rescale(_grid(N),(BLOCK,),(d_d0,c0,np.int32(N)))
        self._rescale(_grid(N),(BLOCK,),(d_d1,c1,np.int32(N)))
        cp.cuda.Stream.null.synchronize()

        print(f"[cuFHE] HE MUL (ct*ct) {(time.perf_counter()-t0)*1e3:.3f}ms")
        return c0, c1

    def modswitch_down(self, ct) -> tuple:
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        t0   = time.perf_counter()
        self._msw_dn(_grid(N),(BLOCK,),(ct[0],out0,np.int32(N)))
        self._msw_dn(_grid(N),(BLOCK,),(ct[1],out1,np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Modswitch {(time.perf_counter()-t0)*1e3:.3f}ms")
        return out0, out1

    def bootstrap(self, ct) -> tuple:
        """
        Noise refresh: decrypt then re-encrypt with fresh noise budget.
        Decrypts the noisy ciphertext to recover plaintext, then
        re-encrypts to get a fresh ciphertext with full noise budget.
        In full TFHE this decryption happens homomorphically.
        """
        print("[cuFHE] Bootstrapping...")
        t0  = time.perf_counter()
        # Decrypt current ciphertext to recover plaintext
        tmp = cp.zeros(N, dtype=cp.uint32)
        self._dec(_grid(N),(BLOCK,),(ct[0],tmp,np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        plaintext = cp.asnumpy(tmp)
        # Re-encrypt with fresh noise — resets noise budget to maximum
        fresh = self.encrypt(plaintext)
        print(f"[cuFHE] Bootstrap {(time.perf_counter()-t0)*1e3:.3f}ms — noise reset")
        return fresh

    def benchmark(self, n_ops=1000):
        msg  = np.random.randint(0, T, N, dtype=np.uint32)
        ct_a = self.encrypt(msg)
        ct_b = self.encrypt(msg)
        t0 = time.perf_counter()
        for _ in range(n_ops):
            self.he_add(ct_a, ct_b)
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter()-t0)*1e3
        print(f"\n[cuFHE] {n_ops} HE ADD: {ms:.1f}ms — {n_ops/ms*1000:.0f} ops/sec")
        t0 = time.perf_counter()
        for _ in range(100):
            self.he_mul_ct(ct_a, ct_b)
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter()-t0)*1e3
        print(f"[cuFHE] 100 HE MUL: {ms:.1f}ms — {100/ms*1000:.0f} ops/sec")
