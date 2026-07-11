"""
========================================================================
Title       : Covariance Matrix and the Uncertainty Condition of Gaussian States
Book        : Quantum Entanglement, Volume 1
Chapter     : 7  --  Continuous Variables and Correlations Beyond Entanglement
Code Ref    : gaussian_state
Description : Builds the covariance matrix of the vacuum and thermal states,
              computes their symplectic eigenvalues, and tests the
              uncertainty condition that certifies a physical Gaussian state.
========================================================================
"""
import numpy as np

# hbar = 1 convention; quadratures obey [x, p] = i
# vacuum convention: covariance matrix of vacuum is (1/2) I

def symplectic_form_single_mode():
    # single-mode symplectic form Omega, ordering (x, p)
    return np.array([[0.0, 1.0],
                     [-1.0, 0.0]])

def cov_vacuum():
    # vacuum state: sigma = (1/2) I
    return 0.5 * np.eye(2)

def cov_thermal(n_bar):
    # thermal state with mean photon number n_bar: sigma = (n_bar + 1/2) I
    return (n_bar + 0.5) * np.eye(2)

def symplectic_eigenvalue_single_mode(sigma):
    # for a single mode the symplectic eigenvalue equals sqrt(det sigma)
    # this follows directly from Williamson's theorem for one mode
    return np.sqrt(np.linalg.det(sigma))

def is_physical(sigma):
    # uncertainty condition: sigma + (i/2) Omega must be positive semidefinite
    Omega = symplectic_form_single_mode()
    M = sigma + 0.5j * Omega
    eigvals = np.linalg.eigvalsh(M)
    min_eig = np.min(eigvals.real)
    return min_eig, (min_eig >= -1e-12)

print("=" * 60)
print("Vacuum state")
sig_vac = cov_vacuum()
print("covariance matrix:")
print(sig_vac)
nu_vac = symplectic_eigenvalue_single_mode(sig_vac)
min_eig_vac, phys_vac = is_physical(sig_vac)
print("symplectic eigenvalue nu = %.6f" % nu_vac)
print("min eigenvalue of (sigma + i/2 Omega) = %.6f" % min_eig_vac)
print("nu >= 1/2 : %s" % (nu_vac >= 0.5 - 1e-12))
print("physical  : %s" % phys_vac)

print("=" * 60)
print("Thermal state for several mean photon numbers")
for n_bar in [0.5, 1.0, 2.0, 5.0]:
    sig_th = cov_thermal(n_bar)
    nu_th = symplectic_eigenvalue_single_mode(sig_th)
    min_eig_th, phys_th = is_physical(sig_th)
    print("n_bar = %.1f" % n_bar)
    print("   nu = %.6f" % nu_th)
    print("   min eigenvalue of (sigma + i/2 Omega) = %.6f" % min_eig_th)
    print("   nu >= 1/2 : %s | physical: %s" % (nu_th >= 0.5 - 1e-12, phys_th))

print("=" * 60)
print("Unphysical test matrix: sigma = 0.3 I (below uncertainty bound)")
sig_bad = 0.3 * np.eye(2)
nu_bad = symplectic_eigenvalue_single_mode(sig_bad)
min_eig_bad, phys_bad = is_physical(sig_bad)
print("   nu = %.6f" % nu_bad)
print("   min eigenvalue of (sigma + i/2 Omega) = %.6f" % min_eig_bad)
print("   nu >= 1/2 : %s | physical: %s" % (nu_bad >= 0.5 - 1e-12, phys_bad))