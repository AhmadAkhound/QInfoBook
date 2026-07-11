"""
========================================================================
Title       : Quantum Fisher Information and Entanglement Depth
Book        : Quantum Entanglement, Volume 1
Chapter     : 5  --  Uncertainty Relations, Witnesses, and Multipartite Entanglement
Code Ref    : quantum_fisher
Description : Computes the quantum Fisher information of GHZ, W, product, and
              noisy GHZ states with respect to a collective spin operator, and
              compares it against the standard quantum limit and the
              Heisenberg limit to certify entanglement depth.
========================================================================
"""
import numpy as np

# single-qubit Pauli operators
I2 = np.eye(2, dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def collective_Jz(n):
    # J_z = (1/2) sum_i sigma_z^(i), exact
    dim = 2**n
    J = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        ops = [I2] * n
        ops[i] = sz
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        J += 0.5 * term
    return J


def qfi_pure(psi, J):
    # pure-state QFI: F_Q = 4 (<J^2> - <J>^2), exact
    mean = np.vdot(psi, J @ psi).real
    mean2 = np.vdot(psi, J @ (J @ psi)).real
    return 4 * (mean2 - mean**2)


def qfi_mixed(rho, J):
    # mixed-state QFI from the eigenbasis: F_Q = 2 sum_{k,l} (p_k-p_l)^2/(p_k+p_l) |<k|J|l>|^2
    vals, vecs = np.linalg.eigh(rho)
    F = 0.0
    for k in range(len(vals)):
        for l in range(len(vals)):
            s = vals[k] + vals[l]
            if s > 1e-12:
                Jkl = np.vdot(vecs[:, k], J @ vecs[:, l])
                F += 2 * (vals[k] - vals[l])**2 / s * abs(Jkl)**2
    return F.real


n = 3
J = collective_Jz(n)

# GHZ state
ghz = np.zeros(2**n, dtype=complex)
ghz[0] = ghz[-1] = 1 / np.sqrt(2)

# W state
w = np.zeros(2**n, dtype=complex)
for i in range(n):
    w[1 << i] = 1 / np.sqrt(n)

# product state |+++>
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
prod = plus
for _ in range(n - 1):
    prod = np.kron(prod, plus)

print("Quantum Fisher information (n = %d qubits)" % n)
print("standard quantum limit N = %d, Heisenberg limit N^2 = %d" % (n, n**2))
print()
print("%-16s | %-8s | %-s" % ("state", "F_Q", "entanglement depth > F_Q/N"))
print("-" * 52)
for name, psi in [("GHZ", ghz), ("W", w), ("product |+++>", prod)]:
    F = qfi_pure(psi, J)
    print("%-16s | %-8.4f | depth > %.2f" % (name, F, F / n))

# noisy GHZ: rho = (1-q) |GHZ><GHZ| + q I/8
print()
print("noisy GHZ:  rho = (1-q)|GHZ><GHZ| + q I/8")
print("%-8s | %-8s | %-s" % ("q", "F_Q", "depth certified"))
print("-" * 40)
P = np.outer(ghz, ghz.conj())
for q in [0.0, 0.2, 0.4, 0.6, 0.8]:
    rho = (1 - q) * P + q * np.eye(2**n) / (2**n)
    F = qfi_mixed(rho, J)
    depth = int(np.floor(F / n)) + 1 if F > n else 1
    print("%-8.1f | %-8.4f | >= %d" % (q, F, depth))