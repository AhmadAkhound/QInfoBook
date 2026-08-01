"""
========================================================================
Title       : Quantum Fisher Information and Entanglement Depth
Book        : Quantum Entanglement, Volume 1
Chapter     : 5 -- Uncertainty Relations, Witnesses, and Multipartite
              Entanglement
Code Ref    : quantum_fisher
Description : Computes the quantum Fisher information of GHZ, W, product,
              and noisy GHZ states with respect to a collective spin
              operator, and applies the exact k-producibility bounds to
              certify multipartite entanglement depth.
========================================================================
"""

import numpy as np


# ---------------------------------------------------------------------
# Single-qubit operators
# ---------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def collective_Jz(n):
    """
    Construct the collective spin operator

        J_z = (1/2) sum_i sigma_z^(i)

    for n qubits.
    """
    dim = 2**n
    J = np.zeros((dim, dim), dtype=complex)

    for i in range(n):
        ops = [I2.copy() for _ in range(n)]
        ops[i] = sz

        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)

        J += 0.5 * term

    return J


def qfi_pure(psi, J, tol=1e-12):
    """
    Quantum Fisher information for a pure state:

        F_Q[|psi>,J] = 4(<J^2> - <J>^2).
    """
    mean = np.vdot(psi, J @ psi).real
    mean2 = np.vdot(psi, J @ (J @ psi)).real
    F = 4.0 * (mean2 - mean**2)

    # Remove tiny negative values caused by floating-point roundoff.
    if abs(F) < tol:
        F = 0.0

    return float(F)


def qfi_mixed(rho, J, tol=1e-12):
    """
    Quantum Fisher information for a mixed state:

        F_Q[rho,J]
        = 2 sum_{k,l}
          (p_k-p_l)^2/(p_k+p_l) |<k|J|l>|^2.
    """
    vals, vecs = np.linalg.eigh(rho)
    F = 0.0

    for k in range(len(vals)):
        for l in range(len(vals)):
            denominator = vals[k] + vals[l]

            if denominator > tol:
                Jkl = np.vdot(vecs[:, k], J @ vecs[:, l])
                F += (
                    2.0
                    * (vals[k] - vals[l])**2
                    / denominator
                    * abs(Jkl)**2
                )

    F = float(np.real(F))

    # Remove tiny negative values caused by floating-point roundoff.
    if abs(F) < tol:
        F = 0.0

    return F


def certified_entanglement_depth(F, n, tol=1e-10):
    """
    Return the largest entanglement depth certified by the
    k-producibility QFI bounds

        F_Q <= s k^2 + r^2,

    where

        n = s k + r,     0 <= r < k.

    A return value of 1 means that this QFI witness does not
    certify entanglement.
    """
    certified_depth = 1

    for k in range(1, n):
        s, r = divmod(n, k)
        k_producible_bound = s * k**2 + r**2

        if F > k_producible_bound + tol:
            certified_depth = k + 1

    return certified_depth


def depth_label(depth):
    """
    Produce an informative output label for the certified depth.
    """
    if depth == 1:
        return "not certified"

    return f">= {depth}"


# ---------------------------------------------------------------------
# States and generator
# ---------------------------------------------------------------------
n = 3
J = collective_Jz(n)

# GHZ state: (|000> + |111>)/sqrt(2)
ghz = np.zeros(2**n, dtype=complex)
ghz[0] = 1.0 / np.sqrt(2)
ghz[-1] = 1.0 / np.sqrt(2)

# W state: (|001> + |010> + |100>)/sqrt(3)
w = np.zeros(2**n, dtype=complex)
for i in range(n):
    w[1 << i] = 1.0 / np.sqrt(n)

# Product state |+++>
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
prod = plus.copy()

for _ in range(n - 1):
    prod = np.kron(prod, plus)


# ---------------------------------------------------------------------
# Pure-state results
# ---------------------------------------------------------------------
print(f"Quantum Fisher information (n = {n} qubits)")
print(
    f"standard quantum limit N = {n}, "
    f"Heisenberg limit N^2 = {n**2}"
)
print()

print("%-16s | %-8s | %-s" %
      ("state", "F_Q", "depth certified"))
print("-" * 52)

pure_states = [
    ("GHZ", ghz),
    ("W", w),
    ("product |+++>", prod),
]

for name, psi in pure_states:
    F = qfi_pure(psi, J)
    depth = certified_entanglement_depth(F, n)

    print(
        "%-16s | %-8.4f | %s"
        % (name, F, depth_label(depth))
    )


# ---------------------------------------------------------------------
# Noisy GHZ states
# rho(q) = (1-q)|GHZ><GHZ| + q I/8
# ---------------------------------------------------------------------
P_ghz = np.outer(ghz, ghz.conj())

print()
print("noisy GHZ: rho = (1-q)|GHZ><GHZ| + q I/8")
print("%-8s | %-8s | %-s" %
      ("q", "F_Q", "depth certified"))
print("-" * 44)

for q in [0.0, 0.2, 0.4, 0.6, 0.8]:
    rho = (
        (1.0 - q) * P_ghz
        + q * np.eye(2**n, dtype=complex) / (2**n)
    )

    F = qfi_mixed(rho, J)
    depth = certified_entanglement_depth(F, n)

    print(
        "%-8.1f | %-8.4f | %s"
        % (q, F, depth_label(depth))
    )