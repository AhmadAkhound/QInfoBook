"""
========================================================================
Title       : Stabilizer Renyi Entropy and Quantum Magic
Book        : Quantum Entanglement, Volume 1
Chapter     : 6  --  Resource Theory and Many-Body Entanglement
Code Ref    : stabilizer_entropy
Description : Computes the second stabilizer Renyi entropy (quantum magic)
              from the Pauli spectrum of one- and two-qubit states, showing
              that stabilizer states have zero magic while T-type states do
              not, and that magic is independent of entanglement.
========================================================================
"""
import numpy as np
from itertools import product

# single-qubit Pauli operators
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
paulis = [I2, X, Y, Z]
labels = ['I', 'X', 'Y', 'Z']


def pauli_group(n):
    # all n-qubit Pauli strings as full matrices, exact
    for idx in product(range(4), repeat=n):
        mat = paulis[idx[0]]
        name = labels[idx[0]]
        for k in idx[1:]:
            mat = np.kron(mat, paulis[k])
            name += labels[k]
        yield name, mat


def stabilizer_renyi_2(rho, n):
    # M2 = -log2( 2^n * sum_P Xi_P^2 ),  Xi_P = (1/2^n) Tr(P rho)^2, exact
    d = 2**n
    total = 0.0
    for _, P in pauli_group(n):
        xi = (np.trace(P @ rho).real)**2 / d
        total += xi**2
    return -np.log2(d * total)


def state_from_vec(psi):
    return np.outer(psi, psi.conj())


# ---------- single-qubit states ----------
print("=== single-qubit magic (no entanglement possible) ===")
zero = np.array([1, 0], dtype=complex)                       # stabilizer
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)          # stabilizer
tstate = np.array([1, np.exp(1j*np.pi/4)], dtype=complex)/np.sqrt(2)  # magic
hstate = np.array([np.cos(np.pi/8), np.sin(np.pi/8)], dtype=complex)  # H-type magic

for name, psi in [("|0> (stabilizer)", zero), ("|+> (stabilizer)", plus),
                  ("T state", tstate), ("H state", hstate)]:
    M2 = stabilizer_renyi_2(state_from_vec(psi), 1)
    print("  %-18s M2 = %.4f" % (name, M2))

# ---------- two-qubit: entanglement vs magic ----------
print()
print("=== two-qubit: entanglement does not imply magic ===")
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)     # max entangled, stabilizer
M2_bell = stabilizer_renyi_2(state_from_vec(bell), 2)
print("  Bell state (max entangled): M2 = %.4f" % M2_bell)

# apply a T gate on the first qubit of the Bell state -> adds magic
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)
TI = np.kron(T, I2)
bell_t = TI @ bell
M2_bell_t = stabilizer_renyi_2(state_from_vec(bell_t), 2)
print("  Bell + one T gate:          M2 = %.4f" % M2_bell_t)

# ---------- magic growth with number of T gates ----------
print()
print("=== magic growth with successive T gates on |+> ===")
print("  %-10s | %-s" % ("n_T gates", "M2"))
print("  " + "-" * 24)
psi = plus.copy()
Tsingle = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)
for k in range(5):
    M2 = stabilizer_renyi_2(state_from_vec(psi), 1)
    print("  %-10d | %.4f" % (k, M2))
    psi = Tsingle @ psi
    psi = psi / np.linalg.norm(psi)