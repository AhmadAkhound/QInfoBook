"""
========================================================================
Title       : Realignment Criterion and Negativity Comparison
Book        : Quantum Entanglement, Volume 1
Chapter     : 3  --  Geometric Measures and the Partial Transpose
Code Ref    : realignment_negativity
Description : Compares the realignment (CCNR) criterion with the negativity
              for detecting entanglement, including states where realignment
              succeeds while the partial transpose does not.
========================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def realignment(rho, dA, dB):
    """
    Realignment of a bipartite density matrix.

    For rho with entries rho_{ij,kl} the realigned matrix is
    R(rho)_{ik,jl} = rho_{ij,kl}, implemented by reshaping the
    four-index tensor and permuting axes.
    """
    R = rho.reshape(dA, dB, dA, dB)
    R = R.transpose(0, 2, 1, 3)
    R = R.reshape(dA * dA, dB * dB)
    return R


def trace_norm(M):
    """Trace norm: sum of singular values of a general matrix."""
    return np.sum(np.linalg.svd(M, compute_uv=False))


def realignment_negativity(rho, dA, dB):
    """Realignment negativity N_R = max(0, ||R(rho)||_1 - 1)."""
    return max(0.0, trace_norm(realignment(rho, dA, dB)) - 1.0)


def partial_transpose(rho, dA, dB):
    """Partial transpose with respect to subsystem B."""
    R = rho.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1)
    return R.reshape(dA * dB, dA * dB)


def ppt_min_eigenvalue(rho, dA, dB):
    """Smallest eigenvalue of the partial transpose."""
    return np.min(np.linalg.eigvalsh(partial_transpose(rho, dA, dB)))


def isotropic_state(F, d=3):
    """
    Isotropic state in d x d parametrized by the fidelity F.

    rho = (1-F)/(d^2-1) (I - P+) + F P+ , with P+ = |Phi+><Phi+|
    and |Phi+> = (1/sqrt(d)) sum_i |ii>.
    """
    phi = np.zeros(d * d, dtype=float)
    for i in range(d):
        phi[i * d + i] = 1.0 / np.sqrt(d)
    P_plus = np.outer(phi, phi)
    I = np.eye(d * d)
    return (1 - F) / (d * d - 1) * (I - P_plus) + F * P_plus


# ---------------------------------------------------
# Numerical table over the fidelity F
# ---------------------------------------------------

F_values = [0.10, 1/3, 0.40, 0.50, 0.70, 0.90, 1.00]

results = []

for F in F_values:

    rho = isotropic_state(F, 3)

    norm_R = trace_norm(realignment(rho, 3, 3))

    n_r = realignment_negativity(rho, 3, 3)

    min_pt = ppt_min_eigenvalue(rho, 3, 3)

    results.append([F, norm_R, n_r, min_pt])

columns = ["F", "||R(rho)||_1", "Realignment Neg.", "Min PT eigenvalue"]

df = pd.DataFrame(results, columns=columns)

print(df.to_string(index=False))
