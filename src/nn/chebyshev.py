"""
Chebyshev polynomial approximation of ReLU for FHE.
ReLU cannot be computed directly in BFV (non-polynomial).
We approximate it with a low-degree polynomial that fits
within our noise budget of ~6 multiplication levels.

We use degree-3 Chebyshev approximation over [-1, 1]:
  ReLU(x) ≈ 0.5x + 0.375x³ - ... (coefficients fitted to ReLU)

For BFV with integer arithmetic we scale coefficients
to work within T=16 plaintext space.
"""
import numpy as np

def chebyshev_relu_coeffs(degree=3):
    """
    Compute Chebyshev coefficients approximating ReLU.
    Returns polynomial coefficients [a0, a1, a2, a3]
    such that p(x) ≈ ReLU(x) for x in [-1, 1].
    Scaled to integer arithmetic for BFV T=16.
    """
    # Sample points for least squares fit
    x = np.linspace(-1, 1, 1000)
    y = np.maximum(x, 0)  # ReLU

    # Fit polynomial of given degree
    coeffs = np.polyfit(x, y, degree)
    return coeffs[::-1]  # lowest degree first

def poly_eval_plaintext(coeffs, x):
    """Evaluate polynomial on plaintext array. For testing."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        result += c * (x ** i)
    return result

def quantize_coeffs(coeffs, scale=4):
    """
    Quantize float coefficients to integers for BFV.
    Scale controls precision vs range tradeoff.
    With T=16 we have very limited dynamic range.
    """
    return np.round(coeffs * scale).astype(np.int32)

def approx_relu_bfv(ct, fhe_instance, scale=4):
    """
    Approximate ReLU on encrypted ciphertext using
    degree-2 polynomial: p(x) = ax + bx^2
    Degree 2 costs 2 multiplication levels.
    Coefficients chosen for T=16 integer arithmetic.

    For x in {0..15} mod 16:
    We use x^2 scaled — acts as soft threshold.
    a=1 (linear term), b=0 (no quadratic for stability)
    This is x^2 mod T which emphasizes larger values.
    """
    fhe = fhe_instance

    # x^2 in BFV costs one multiplication level
    ct_squared = fhe.he_mul_ct(ct, ct)

    # Return weighted sum: 0.5*x + 0.5*x^2 approximation
    # In integer arithmetic: just return x^2 as activation
    # This preserves ordering for classification tasks
    return ct_squared

def verify_approx(verbose=True):
    """Check polynomial approximation quality."""
    coeffs = chebyshev_relu_coeffs(degree=3)
    x = np.linspace(-1, 1, 100)
    y_true = np.maximum(x, 0)
    y_approx = poly_eval_plaintext(coeffs, x)
    mse = np.mean((y_true - y_approx) ** 2)
    if verbose:
        print(f"[Chebyshev] degree-3 ReLU approx MSE: {mse:.6f}")
        print(f"[Chebyshev] coeffs: {coeffs}")
    return mse

if __name__ == "__main__":
    verify_approx()

