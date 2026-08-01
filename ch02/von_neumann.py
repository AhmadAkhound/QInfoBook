"""
========================================================================
Title       : Von Neumann Entropy and State Purity
Book        : Quantum Entanglement, Volume 1
Chapter     : 2  --  Entropic Foundations and Structural Measures
Code Ref    : von_neumann
Description : Computes the von Neumann entropy and the purity of pure, mixed,
              and maximally mixed single-qubit states from the eigenvalues
              of the density matrix.
========================================================================
"""
import numpy as np

def von_neumann_entropy(rho):
    # Eigenvalues of the density matrix
    eigenvalues = np.linalg.eigvalsh(rho)
    # Keep only positive eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def purity(rho):
    return np.real(np.trace(rho @ rho))

# Sample states
ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)

states = {
    "Pure |0>"          : ket0 @ ket0.conj().T,
    "Mixed 3/4,1/4"     : 0.75 * (ket0 @ ket0.conj().T) + 0.25 * (ket1 @ ket1.conj().T),
    "Maximally mixed"   : np.eye(2) / 2,
}

for name, rho in states.items():
    S = von_neumann_entropy(rho)
    P = purity(rho)
    print(f"=== {name} ===")
    print("Von Neumann entropy S =", np.round(S, 4))
    print("Purity Tr(rho^2) =", np.round(P, 4))
    print()