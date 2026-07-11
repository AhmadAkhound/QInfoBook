"""
========================================================================
Title       : Single-Qubit Depolarizing Channel
Book        : Quantum Entanglement, Volume 1
Chapter     : 1  --  Hilbert Space Foundations and Mathematical Tools
Code Ref    : depolarizing_channel
Description : Applies the single-qubit depolarizing channel through its Kraus
              operators, verifies the trace-preserving condition, and tracks
              the output-state purity as the depolarizing strength varies.
========================================================================
"""
import numpy as np

# Pauli matrices and identity
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kraus_depolarizing(p):
    # Kraus operators of the single-qubit depolarizing channel
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p / 3) * X
    K2 = np.sqrt(p / 3) * Y
    K3 = np.sqrt(p / 3) * Z
    return [K0, K1, K2, K3]

def apply_channel(rho, kraus):
    # Apply a quantum channel given its Kraus operators
    return sum(K @ rho @ K.conj().T for K in kraus)

def trace_preserving_check(kraus):
    # Verify sum_i K_i^dagger K_i = I
    return sum(K.conj().T @ K for K in kraus)

# Input pure state |0>
ket0 = np.array([[1], [0]], dtype=complex)
rho_in = ket0 @ ket0.conj().T

for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    kraus = kraus_depolarizing(p)
    rho_out = apply_channel(rho_in, kraus)
    tp = trace_preserving_check(kraus)
    purity = np.trace(rho_out @ rho_out).real
    print(f"=== p = {p} ===")
    print("Output state rho_out =")
    print(np.round(rho_out, 4))
    print("Sum K^dag K =")
    print(np.round(tp, 4))
    print("Purity Tr(rho_out^2) =", np.round(purity, 4))
    print()