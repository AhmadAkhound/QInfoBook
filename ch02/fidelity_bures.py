"""
========================================================================
Title       : Uhlmann-Jozsa Fidelity and Bures Distance
Book        : Quantum Entanglement, Volume 1
Chapter     : 2  --  Entropic Foundations and Structural Measures
Code Ref    : fidelity_bures
Description : Computes the Uhlmann-Jozsa fidelity and the Bures distance
              between a noisy Bell state and the ideal state, and checks the
              overlap of two pure single-qubit states.
========================================================================
"""
import numpy as np
from scipy.linalg import sqrtm

def fidelity(rho, sigma):
    # Uhlmann-Jozsa fidelity F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
    sqrt_rho = sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = sqrtm(inner)
    F = np.real(np.trace(sqrt_inner))**2
    return np.clip(F, 0, 1)  # keep within physical range

def bures_distance(rho, sigma):
    F = fidelity(rho, sigma)
    val = 2 * (1 - np.sqrt(F))
    val = max(val, 0)  # remove tiny negative numerical noise
    return np.sqrt(val)

# Ideal Bell state
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(bell, bell.conj())

# Basis state |01>
k01 = np.array([0, 1, 0, 0], dtype=complex)
rho_01 = np.outer(k01, k01.conj())

print("=== Fidelity of noisy Bell state vs ideal ===")
print(f"{'epsilon':<10} | {'Fidelity':<10} | {'Bures dist':<10}")
print("-" * 36)
for eps in [0.0, 0.1, 0.25, 0.5, 1.0]:
    rho_noisy = (1 - eps) * rho_bell + eps * rho_01
    F = fidelity(rho_noisy, rho_bell)
    dB = bures_distance(rho_noisy, rho_bell)
    print(f"{eps:<10.2f} | {F:<10.4f} | {dB:<10.4f}")

# Two pure states overlap check
print()
print("=== Two pure states ===")
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
zero = np.array([1, 0], dtype=complex)
rho_plus = np.outer(plus, plus.conj())
rho_zero = np.outer(zero, zero.conj())
print("F(|+>, |0>) =", np.round(fidelity(rho_plus, rho_zero), 4), "(expected 0.5)")