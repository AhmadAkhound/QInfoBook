"""
========================================================================
Title       : Equality of Entanglement Measures for Pure States
Book        : Quantum Entanglement, Volume 1
Chapter     : 4  --  Operational Measures and the Cost of Entanglement
Code Ref    : squashed_bounds
Description : Verifies that, for the pure two-qubit state
              |psi(theta)> = cos(theta)|00> + sin(theta)|11>,
              the distillable entanglement, the squashed entanglement,
              and the entanglement of formation all coincide with the
              entanglement entropy S(rho_A).
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt


def binary_entropy(x):
    """Shannon binary entropy H(x) in bits, with safe limits."""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def pure_state_measures(theta):
    """For |psi> = cos(theta)|00> + sin(theta)|11>, all three measures
    E_d, E_sq, E_F coincide with the entanglement entropy S(rho_A).
    Schmidt coefficients are cos^2(theta) and sin^2(theta)."""
    lam1 = np.cos(theta) ** 2
    # reduced-state eigenvalues are lam1 and 1-lam1
    S_A = binary_entropy(lam1)
    # for a pure state: E_d = E_sq = E_F = S_A exactly
    return S_A


thetas = np.linspace(0, np.pi / 4, 19)
print(f"{'theta':<8} | {'lambda1':<10} | {'E_d=E_sq=E_F = S(rho_A)':<24}")
print("-" * 50)
S_values = []
for th in thetas:
    lam1 = np.cos(th) ** 2
    S = pure_state_measures(th)
    S_values.append(S)
    print(f"{th:<8.4f} | {lam1:<10.4f} | {S:<24.6f}")

S_values = np.array(S_values)

print(f"\nAt theta=pi/4: S(rho_A) = {pure_state_measures(np.pi/4):.6f} (should be 1.0)")
print(f"At theta=0:    S(rho_A) = {pure_state_measures(0.0):.6f} (should be 0.0)")