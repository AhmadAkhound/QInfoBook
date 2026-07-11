"""
========================================================================
Title       : Relative Entropy and Quantum Mutual Information
Book        : Quantum Entanglement, Volume 1
Chapter     : 2  --  Entropic Foundations and Structural Measures
Code Ref    : rel_mutual
Description : Computes the quantum relative entropy between two states,
              illustrating its asymmetry, and the mutual information of Bell
              and product states from the von Neumann entropies.
========================================================================
"""
import numpy as np
from scipy.linalg import logm

def relative_entropy(rho, sigma):
    # S(rho || sigma) = Tr(rho log rho) - Tr(rho log sigma), in bits
    log_rho = logm(rho) / np.log(2)
    log_sigma = logm(sigma) / np.log(2)
    return np.real(np.trace(rho @ log_rho) - np.trace(rho @ log_sigma))

def von_neumann(rho):
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-12]
    return -np.sum(eigs * np.log2(eigs))

def partial_trace_B(rho, dA, dB):
    rho = rho.reshape(dA, dB, dA, dB)
    return np.trace(rho, axis1=1, axis2=3)

def partial_trace_A(rho, dA, dB):
    rho = rho.reshape(dA, dB, dA, dB)
    return np.trace(rho, axis1=0, axis2=2)

def mutual_information(rho_AB, dA, dB):
    rho_A = partial_trace_B(rho_AB, dA, dB)
    rho_B = partial_trace_A(rho_AB, dA, dB)
    return von_neumann(rho_A) + von_neumann(rho_B) - von_neumann(rho_AB)

# Part 1: relative entropy between two single-qubit states
rho = np.array([[0.75, 0], [0, 0.25]], dtype=complex)
sigma = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
print("=== Relative entropy ===")
print("S(rho||sigma) =", np.round(relative_entropy(rho, sigma), 4))
print("S(sigma||rho) =", np.round(relative_entropy(sigma, rho), 4))
print("(asymmetry shows it is not a true distance)")
print()

# Part 2: mutual information of the Bell state
k0 = np.array([1, 0], dtype=complex)
k1 = np.array([0, 1], dtype=complex)
bell = (np.kron(k0, k0) + np.kron(k1, k1)) / np.sqrt(2)
rho_bell = np.outer(bell, bell.conj())
print("=== Mutual information ===")
print("I(A:B) for Bell state =", np.round(mutual_information(rho_bell, 2, 2), 4))

# Product state for comparison
prod = np.kron(k0, k1)
rho_prod = np.outer(prod, prod.conj())
print("I(A:B) for product state =", np.round(mutual_information(rho_prod, 2, 2), 4))