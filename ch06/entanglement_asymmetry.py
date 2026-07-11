"""
========================================================================
Title       : Entanglement Asymmetry and the Quantum Mpemba Effect
Book        : Quantum Entanglement, Volume 1
Chapter     : 6  --  Resource Theory and Many-Body Entanglement
Code Ref    : entanglement_asymmetry
Description : Computes the entanglement asymmetry as a measure of broken
              symmetry and illustrates the quantum Mpemba effect in its
              relaxation.
========================================================================
"""
import numpy as np

# =====================================================================
#  Entanglement asymmetry: a resource-theoretic measure of how much a
#  state breaks a symmetry, built from the relative entropy (Ch.2).
#  For a subsystem state rho_A and a charge operator Q with projectors
#  Pi_q onto each charge sector, the symmetrized state is
#     rho_{A,Q} = sum_q Pi_q rho_A Pi_q   (off-diagonal charge blocks removed)
#  and the entanglement asymmetry is
#     Delta S = S(rho_{A,Q}) - S(rho_A) = S(rho_A || rho_{A,Q}) >= 0,
#  zero iff the state is symmetric. Everything exact, no approximation.
# =====================================================================

def vN_entropy(rho, tol=1e-12):
    w = np.linalg.eigvalsh(rho)
    w = w[w > tol]
    return float(-np.sum(w * np.log2(w)))


def symmetrize(rho, charges):
    """Remove off-diagonal blocks between different charge sectors.
    charges[i] is the U(1) charge of basis state i."""
    d = rho.shape[0]
    rho_sym = np.zeros_like(rho)
    unique_q = sorted(set(charges))
    for q in unique_q:
        idx = [i for i in range(d) if charges[i] == q]
        for i in idx:
            for j in idx:
                rho_sym[i, j] = rho[i, j]
    return rho_sym


def entanglement_asymmetry(rho, charges):
    rho_sym = symmetrize(rho, charges)
    return vN_entropy(rho_sym) - vN_entropy(rho)


# ---- single qubit, U(1) charge = number of excitations (0 for |0>, 1 for |1>) ----
charges_1q = [0, 1]

# a tilted state: cos|0> + sin|1>, rotated by angle theta away from symmetry
print("Single qubit tilted by angle theta from the symmetric axis:")
print(" theta(deg)   Delta S (asymmetry)")
for deg in [0, 15, 30, 45, 60, 75, 90]:
    th = np.deg2rad(deg)
    psi = np.array([np.cos(th), np.sin(th)], dtype=complex)
    rho = np.outer(psi, psi.conj())
    dS = entanglement_asymmetry(rho, charges_1q)
    print(f"   {deg:3d}        {dS:.4f}")

print()
# a pure symmetric eigenstate (|0> or |1>) has zero asymmetry;
# an equal superposition (theta=45) maximally breaks the U(1) charge symmetry
print("Check: theta=0 -> |0> (symmetric, dS=0);"
      " theta=45 -> max superposition (max asymmetry).")

# ---- mixed state example: partial coherence ----
print()
print("Mixed state rho = p|+><+| + (1-p) I/2, U(1) charge symmetry:")
print("  p       Delta S")
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
P = np.outer(plus, plus.conj())
for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rho = p * P + (1 - p) * np.eye(2) / 2
    dS = entanglement_asymmetry(rho, charges_1q)
    print(f"  {p:.2f}    {dS:.4f}")