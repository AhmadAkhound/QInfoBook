"""
========================================================================
Title       : Horodecki Bound and Bound Entanglement
Book        : Quantum Entanglement, Volume 1
Chapter     : 3  --  Geometric Measures and the Partial Transpose
Code Ref    : horodecki_bound
Description : Evaluates the Horodecki positive-partial-transpose condition and
              illustrates a bound entangled state that is PPT yet not
              separable.
========================================================================
"""
import numpy as np
from numpy import linalg as LA


def partial_transpose(rho, dims, subsystem=2):
    """Partial transpose of a bipartite density matrix."""
    d1, d2 = dims
    reshaped = rho.reshape([d1, d2, d1, d2])
    if subsystem == 1:
        transposed = reshaped.transpose([2, 1, 0, 3])
    else:
        transposed = reshaped.transpose([0, 3, 2, 1])
    return transposed.reshape([d1 * d2, d1 * d2])


def realignment(rho, dA, dB):
    """Realignment R(rho)_{ik,jl} = rho_{ij,kl}."""
    R = rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3)
    return R.reshape(dA * dA, dB * dB)


def trace_norm(M):
    """Trace norm: sum of singular values."""
    return np.sum(np.linalg.svd(M, compute_uv=False))


# ---------------------------------------------------
# Construct the 3x3 Tiles state from an unextendible
# product basis (UPB). The bound-entangled state is the
# normalized projector onto the complement of the UPB.
# ---------------------------------------------------

e0, e1, e2 = np.eye(3)

upb = [
    np.outer(e0, (e0 - e1) / np.sqrt(2)),
    np.outer((e0 - e1) / np.sqrt(2), e2),
    np.outer(e2, (e1 - e2) / np.sqrt(2)),
    np.outer((e1 - e2) / np.sqrt(2), e0),
    np.outer((e0 + e1 + e2) / np.sqrt(3), (e0 + e1 + e2) / np.sqrt(3)),
]

projection_upb = sum(np.outer(p.flatten(), np.conj(p.flatten())) for p in upb)

rho_bound = (np.eye(9) - projection_upb) / 4.0


# ---------------------------------------------------
# Spectrum of rho and of its partial transpose
# ---------------------------------------------------

eig_rho = LA.eigvalsh(rho_bound)
eig_pt = LA.eigvalsh(partial_transpose(rho_bound, (3, 3)))

print("Index | Eig_Rho   | Eig_PT")
for i in range(9):
    print(f"{i+1:<5} | {eig_rho[i]:<9.4f} | {eig_pt[i]:<9.4f}")

print()
print("Smallest eigenvalue of partial transpose:", np.min(eig_pt))


# ---------------------------------------------------
# Two complementary criteria
# ---------------------------------------------------

min_pt = np.min(eig_pt)
ppt = min_pt >= -1e-10

norm_R = trace_norm(realignment(rho_bound, 3, 3))
realign_neg = max(0.0, norm_R - 1.0)

print()
print("PPT criterion (min PT eigenvalue >= 0):", ppt)
print("Realigned trace norm ||R(rho)||_1:", round(norm_R, 4))
print("Realignment negativity:", round(realign_neg, 4))