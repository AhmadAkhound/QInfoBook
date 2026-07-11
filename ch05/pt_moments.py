"""
========================================================================
Title       : Entanglement Detection from Partial-Transpose Moments
Book        : Quantum Entanglement, Volume 1
Chapter     : 5  --  Uncertainty Relations, Witnesses, and Multipartite Entanglement
Code Ref    : pt_moments
Description : Detects entanglement from the moments of the partially transposed
              density matrix, using the p3-PPT condition that avoids full state
              tomography.
========================================================================
"""
import numpy as np

# =====================================================================
#  Entanglement detection from partial-transpose (PT) moments — the
#  p3-PPT criterion (Elben et al., PRL 125, 200501 (2020)). PT moments
#  p_k = Tr[(rho^{T_B})^k] can be estimated from local randomized
#  measurements WITHOUT full tomography. The p3-PPT criterion states:
#  every PPT state satisfies p3 >= p2^2; hence p3 < p2^2 certifies
#  entanglement. Everything exact from first principles.
# =====================================================================

phi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
bell = np.outer(phi, phi.conj())


def werner(v):
    return v * bell + (1 - v) * np.eye(4) / 4


def partial_transpose(rho, dA, dB):
    r = rho.reshape(dA, dB, dA, dB)
    return r.transpose(0, 3, 2, 1).reshape(dA * dB, dA * dB)


def pt_moment(rho, dA, dB, k):
    """k-th PT moment p_k = Tr[(rho^{T_B})^k], exact."""
    rTB = partial_transpose(rho, dA, dB)
    M = np.linalg.matrix_power(rTB, k)
    return float(np.real(np.trace(M)))


print("Werner state: p3-PPT entanglement criterion")
print(" v       p2        p3        p2^2      p3<p2^2 (entangled)?   min eig PT")
for v in [0.30, 1/3, 0.40, 0.50, 0.60, 0.707, 0.80, 1.0]:
    rho = werner(v)
    p2 = pt_moment(rho, 2, 2, 2)
    p3 = pt_moment(rho, 2, 2, 3)
    rTB = partial_transpose(rho, 2, 2)
    min_eig = np.min(np.linalg.eigvalsh(rTB))
    detected = p3 < p2**2 - 1e-12
    print(f" {v:.3f}   {p2:.4f}   {p3:.4f}    {p2**2:.4f}      "
          f"{str(detected):>5}                {min_eig:+.4f}")

print()
print("p3-PPT detects entanglement when p3 < p2^2.")
print("Compare with exact PPT (negative eigenvalue of partial transpose):")
print("both should agree that Werner is entangled for v > 1/3.")