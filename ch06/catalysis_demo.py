"""
========================================================================
Title       : Entanglement Catalysis and Majorization
Book        : Quantum Entanglement, Volume 1
Chapter     : 6  --  Resource Theory and Many-Body Entanglement
Code Ref    : catalysis_demo
Description : Demonstrates entanglement catalysis, where a transformation
              forbidden by majorization becomes possible in the presence of a
              catalyst state that is returned unchanged.
========================================================================
"""
import numpy as np

def majorizes(lam_source, lam_target, tol=1e-12):
    """Check whether lam_target majorizes lam_source (lam_source < lam_target)."""
    n = max(len(lam_source), len(lam_target))
    s = np.sort(np.pad(lam_source, (0, n - len(lam_source))))[::-1]
    t = np.sort(np.pad(lam_target, (0, n - len(lam_target))))[::-1]
    ps, pt = np.cumsum(s), np.cumsum(t)
    return np.all(ps <= pt + tol), ps, pt


# Jonathan-Plenio catalysis example
lam_psi = np.array([0.4, 0.4, 0.1, 0.1])
lam_phi = np.array([0.5, 0.25, 0.25, 0.0])
lam_c   = np.array([0.6, 0.4])

# direct conversion psi -> phi requires lam_psi < lam_phi
ok_direct, ps, pt = majorizes(lam_psi, lam_phi)
print("Direct  psi -> phi : majorization holds?", ok_direct)
print("  partial sums psi:", np.round(ps, 4))
print("  partial sums phi:", np.round(pt, 4))
print()

# catalysed conversion: outer product with the catalyst
lam_psi_c = np.outer(lam_psi, lam_c).flatten()
lam_phi_c = np.outer(lam_phi, lam_c).flatten()
ok_cat, psc, ptc = majorizes(lam_psi_c, lam_phi_c)
print("Catalysed psi(x)c -> phi(x)c : majorization holds?", ok_cat)
print("  sorted psi(x)c:", np.round(np.sort(lam_psi_c)[::-1], 4))
print("  sorted phi(x)c:", np.round(np.sort(lam_phi_c)[::-1], 4))
print("  partial sums psi(x)c:", np.round(psc, 4))
print("  partial sums phi(x)c:", np.round(ptc, 4))
print()

# catalyst unchanged before and after
print("Catalyst lam_c before and after:", lam_c, "(returned intact)")

# entanglement entropies, for confirmation
def entropy(lam, tol=1e-12):
    p = np.asarray(lam)
    p = p[p > tol]
    return float(-np.sum(p * np.log2(p)))

print()
print("E(psi)      =", round(entropy(lam_psi), 4))
print("E(phi)      =", round(entropy(lam_phi), 4))