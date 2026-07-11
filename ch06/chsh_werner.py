"""
========================================================================
Title       : CHSH Inequality and the Correlation Hierarchy of the Werner State
Book        : Quantum Entanglement, Volume 1
Chapter     : 6  --  Resource Theory and Many-Body Entanglement
Code Ref    : chsh_werner
Description : Evaluates the maximal CHSH value of the Werner state and locates
              the three thresholds that separate Bell nonlocality, steering,
              and entanglement.
========================================================================
"""
import numpy as np

# =====================================================================
#  Three-layer hierarchy of quantum correlations for the Werner state
#    rho(v) = v |Phi+><Phi+| + (1-v) I/4
#  Computed from first principles:
#   - entanglement threshold v > 1/3   (concurrence, from Ch.2)
#   - steering threshold     v > 1/2   (projective measurements)
#   - CHSH nonlocality       v > 1/sqrt(2)  (optimal CHSH value)
#  The optimal CHSH value of a two-qubit state follows from the
#  Horodecki criterion: S_max = 2 sqrt(t1^2 + t2^2), where t1>=t2 are
#  the two largest singular values of the 3x3 correlation matrix T.
# =====================================================================

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
paulis = [X, Y, Z]

# Bell state |Phi+>
phi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
bell = np.outer(phi, phi.conj())


def werner(v):
    return v * bell + (1 - v) * np.eye(4) / 4


def correlation_matrix(rho):
    """T_ij = Tr[rho (sigma_i x sigma_j)], exact."""
    T = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            op = np.kron(paulis[i], paulis[j])
            T[i, j] = np.real(np.trace(rho @ op))
    return T


def concurrence(rho):
    """Wootters concurrence, exact."""
    YY = np.kron(Y, Y)
    rho_tilde = YY @ rho.conj() @ YY
    R = rho @ rho_tilde
    ev = np.sort(np.sqrt(np.clip(np.linalg.eigvals(R).real, 0, None)))[::-1]
    return max(0.0, ev[0] - ev[1] - ev[2] - ev[3])


def chsh_max(rho):
    """Horodecki optimal CHSH value: 2 sqrt(t1^2 + t2^2)."""
    T = correlation_matrix(rho)
    s = np.sort(np.linalg.svd(T, compute_uv=False))[::-1]
    return 2 * np.sqrt(s[0]**2 + s[1]**2)


print(" v      concurrence   CHSH_max   entangled  steerable  nonlocal")
for v in [0.30, 1/3, 0.40, 0.50, 0.60, 1/np.sqrt(2), 0.75, 0.90, 1.0]:
    rho = werner(v)
    C = concurrence(rho)
    S = chsh_max(rho)
    ent = "yes" if v > 1/3 + 1e-9 else "no"
    steer = "yes" if v > 0.5 + 1e-9 else "no"
    nonloc = "yes" if S > 2 + 1e-9 else "no"
    print(f" {v:.3f}   {C:+.4f}      {S:.4f}    {ent:>5}      {steer:>5}     {nonloc:>5}")

print()
print("Thresholds:  entanglement v>1/3=%.3f,  steering v>1/2=0.500,"
      "  nonlocality v>1/sqrt2=%.3f" % (1/3, 1/np.sqrt(2)))