"""
========================================================================
Title       : Entanglement Entropy via Schmidt Decomposition
Book        : Quantum Entanglement, Volume 1
Chapter     : 2  --  Entropic Foundations and Structural Measures
Code Ref    : ent_entropy
Description : Computes the entanglement entropy of bipartite pure states
              from the squared Schmidt coefficients obtained by singular
              value decomposition of the state matrix.
========================================================================
"""
import numpy as np

def entanglement_entropy(psi, dim_A, dim_B):
    # Reshape state vector into a matrix and use SVD
    M = psi.reshape(dim_A, dim_B)
    schmidt_coeffs = np.linalg.svd(M, compute_uv=False)
    # Squared Schmidt coefficients are eigenvalues of reduced state
    lambdas = schmidt_coeffs**2
    lambdas = lambdas[lambdas > 1e-12]
    return -np.sum(lambdas * np.log2(lambdas))

# Basis kets
k0 = np.array([1, 0], dtype=complex)
k1 = np.array([0, 1], dtype=complex)

# Sample pure states (two qubits)
states = {
    "Product |00>"          : np.kron(k0, k0),
    "Bell (|00>+|11>)/v2"   : (np.kron(k0, k0) + np.kron(k1, k1)) / np.sqrt(2),
    "Partial v0.8|00>+v0.2|11>": np.sqrt(0.8)*np.kron(k0, k0) + np.sqrt(0.2)*np.kron(k1, k1),
}

for name, psi in states.items():
    E = entanglement_entropy(psi, 2, 2)
    print(f"=== {name} ===")
    print("Entanglement entropy E =", np.round(E, 4))
    print()