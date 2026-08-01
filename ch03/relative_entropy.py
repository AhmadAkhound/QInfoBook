"""
========================================================================
Title       : Relative Entropy of Entanglement of the Werner State
Book        : Quantum Entanglement, Volume 1
Chapter     : 3  --  Geometric Measures and the Partial Transpose
Code Ref    : relative_entropy
Description : Computes the relative entropy of entanglement of the Werner
              state with respect to the closest separable state as a function
              of the mixing parameter.
========================================================================
"""
import numpy as np
import pandas as pd


def binary_entropy(x):
    """Binary Shannon entropy H(x) = -x log2 x - (1-x) log2(1-x)."""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def xlog2_ratio(a, b):
    """Safe evaluation of a * log2(a/b) with the limit 0*log2(0)=0."""
    if a <= 0:
        return 0.0
    return a * np.log2(a / b)


def ree_pure(theta):
    """
    Relative entropy of entanglement for the pure state
    cos(theta)|00> + sin(theta)|11>. For a pure state it equals
    the entanglement entropy S(rho_A), the binary entropy of the
    squared Schmidt coefficients.
    """
    return binary_entropy(np.cos(theta) ** 2)


def ree_werner(p):
    """
    Relative entropy of entanglement for the Werner state in the
    entangled region (p > 1/3). The closest separable state is the
    boundary Werner state at p = 1/3.
    """
    if p <= 1 / 3:
        return 0.0

    lam_s = (1 + 3 * p) / 4      # eigenvalue on the singlet
    lam_t = (1 - p) / 4          # eigenvalue on each triplet
    mu_s = 0.5                   # reference eigenvalue on the singlet
    mu_t = 1 / 6                 # reference eigenvalue on each triplet

    term_s = xlog2_ratio(lam_s, mu_s)
    term_t = 3 * xlog2_ratio(lam_t, mu_t)

    return term_s + term_t


# ---------------------------------------------------
# Table for the pure state versus Schmidt angle
# ---------------------------------------------------

theta_deg = [0.0, 22.5, 30.0, 45.0, 60.0, 90.0]

rows_pure = [[t, ree_pure(np.deg2rad(t))] for t in theta_deg]

df_pure = pd.DataFrame(rows_pure, columns=["theta (deg)", "E_R (pure)"])

print("Pure state cos(theta)|00> + sin(theta)|11>:")
print(df_pure.to_string(index=False))
print()


# ---------------------------------------------------
# Table for the Werner state versus mixing parameter
# ---------------------------------------------------

p_vals = [0.0, 0.2, 1/3, 0.4, 0.6, 0.8, 1.0]

rows_w = [[p, ree_werner(p)] for p in p_vals]

df_w = pd.DataFrame(rows_w, columns=["p", "E_R (Werner)"])

print("Werner state:")
print(df_w.to_string(index=False))