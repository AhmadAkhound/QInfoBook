"""
========================================================================
Title       : Concurrence of Bell, Product, and Werner States
Book        : Quantum Entanglement, Volume 1
Chapter     : 2  --  Entropic Foundations and Structural Measures
Code Ref    : concurrence_werner
Description : Computes the concurrence via the spin-flip operator for Bell,
              product, and Werner states, showing the vanishing of
              entanglement below the separability threshold.
========================================================================
"""
import numpy as np

# Pauli-y operator
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Y_tensor = np.kron(sigma_y, sigma_y)

def concurrence(rho):
    # Spin-flipped state: rho_tilde = (sy x sy) rho* (sy x sy)
    rho_tilde = Y_tensor @ rho.conj() @ Y_tensor
    # R = rho * rho_tilde
    R = rho @ rho_tilde
    # Eigenvalues of R (real, non-negative up to numerical error)
    eigs = np.linalg.eigvals(R)
    eigs = np.real(eigs)
    eigs = np.sort(eigs)[::-1]  # descending order
    eigs = np.clip(eigs, 0, None)  # remove small negative noise
    # Square roots of eigenvalues
    sqrt_eigs = np.sqrt(eigs)
    C = sqrt_eigs[0] - sqrt_eigs[1] - sqrt_eigs[2] - sqrt_eigs[3]
    return max(0, C)

# Bell state
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(bell, bell.conj())

# Product state
prod = np.array([1, 0, 0, 0], dtype=complex)
rho_prod = np.outer(prod, prod.conj())

# Werner state at several mixing parameters
def werner(p):
    return p * rho_bell + (1 - p) * np.eye(4) / 4

print("=== Concurrence tests ===")
print("Bell state    C =", np.round(concurrence(rho_bell), 4))
print("Product state C =", np.round(concurrence(rho_prod), 4))
print()
print("=== Werner state ===")
for p in [0.2, 1/3, 0.5, 0.8, 1.0]:
    C = concurrence(werner(p))
    print(f"p = {p:.3f}  ->  C = {np.round(C, 4)}")