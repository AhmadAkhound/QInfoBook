"""
========================================================================
Title       : Entropic Uncertainty Relations and the Berta Bound
Book        : Quantum Entanglement, Volume 1
Chapter     : 5  --  Uncertainty Relations, Witnesses, and Multipartite Entanglement
Code Ref    : entropic_uncertainty
Description : Computes the entropic uncertainty bound with quantum memory and
              its tightening by entanglement, comparing the measured entropies
              against the Berta lower bound.
========================================================================
"""
import numpy as np
from numpy import log2

# ---------- pauli operators and bell state ----------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
P_phi = np.outer(phi_plus, phi_plus.conj())          # |Phi+><Phi+|

def werner(p):
    """Werner state: p |Phi+><Phi+| + (1-p) I/4 (eq. ch2 werner def)."""
    return p * P_phi + (1 - p) * np.eye(4, dtype=complex) / 4

# ---------- entropy from eigenvalues (von Neumann) ----------
def vn_entropy(rho):
    """von Neumann entropy in bits, built from eigenvalues directly."""
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]                              # drop numerical zeros
    return float(-np.sum(ev * log2(ev)))

# ---------- partial trace over subsystem B (2x2 blocks) ----------
def partial_trace_B(rho):
    r = rho.reshape(2, 2, 2, 2)
    return np.trace(r, axis1=1, axis2=3)            # rho_A

def partial_trace_A(rho):
    r = rho.reshape(2, 2, 2, 2)
    return np.trace(r, axis1=0, axis2=2)            # rho_B
    # ---------- Holevo quantity for a local measurement on A ----------
def holevo_X_B(rho_AB, M):
    """
    Holevo information I(X:B) for measuring observable M on subsystem A.
    Built from the definition: I(X:B) = S(rho_B) - sum_i p_i S(rho_B^{x_i}).
    M is a 2x2 Hermitian observable on A; its eigenvectors define the basis.
    """
    rho_B = partial_trace_A(rho_AB)
    S_B = vn_entropy(rho_B)

    # eigenvectors of M give the measurement basis on A
    _, vecs = np.linalg.eigh(M)
    cond_term = 0.0
    for i in range(2):
        proj_A = np.outer(vecs[:, i], vecs[:, i].conj())     # |x_i><x_i| on A
        P = np.kron(proj_A, I2)                              # acts on A only
        rho_post = P @ rho_AB @ P
        p_i = float(np.real(np.trace(rho_post)))
        if p_i > 1e-12:
            rho_post = rho_post / p_i
            rho_B_cond = partial_trace_A(rho_post)           # rho_B^{x_i}
            cond_term += p_i * vn_entropy(rho_B_cond)
    return S_B - cond_term

# ---------- bounds and correction term ----------
def uncertainty_report(p):
    rho = werner(p)
    S_AB = vn_entropy(rho)
    S_A = vn_entropy(partial_trace_B(rho))
    S_B = vn_entropy(partial_trace_A(rho))

    cond_AB = S_AB - S_B                 # S(A|B)
    mutual  = S_A + S_B - S_AB           # I(A:B)

    log_term = log2(2.0)                 # log2(1/c), c = 1/2  -> 1 bit

    mu_bound = log_term                              # Maassen-Uffink
    berta_bound = log_term + cond_AB                 # Berta

    I_xB = holevo_X_B(rho, sx)
    I_zB = holevo_X_B(rho, sz)
    delta = mutual - (I_xB + I_zB)
    adabi_bound = berta_bound + max(0.0, delta)      # Adabi

    return dict(p=p, S_AB=S_AB, cond_AB=cond_AB, mutual=mutual,
                I_xB=I_xB, I_zB=I_zB, delta=delta,
                MU=mu_bound, Berta=berta_bound, Adabi=adabi_bound)

# ---------- table over a range of p ----------
print(f"{'p':>5} {'S(AB)':>8} {'S(A|B)':>8} {'I(A:B)':>8} "
      f"{'delta':>8} {'MU':>6} {'Berta':>8} {'Adabi':>8}")
for p in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
    r = uncertainty_report(p)
    print(f"{r['p']:>5.2f} {r['S_AB']:>8.4f} {r['cond_AB']:>8.4f} "
          f"{r['mutual']:>8.4f} {r['delta']:>8.4f} {r['MU']:>6.2f} "
          f"{r['Berta']:>8.4f} {r['Adabi']:>8.4f}")
    # ---------- figure: hierarchy of the three uncertainty bounds ----------
import matplotlib.pyplot as plt

p_vals = np.linspace(0.30, 1.0, 71)
mu_list, berta_list, adabi_list = [], [], []
for p in p_vals:
    r = uncertainty_report(p)
    mu_list.append(r['MU'])
    berta_list.append(r['Berta'])
    adabi_list.append(r['Adabi'])

plt.figure(figsize=(7, 5))
plt.plot(p_vals, mu_list, label='Maassen-Uffink', linestyle='--', linewidth=2)
plt.plot(p_vals, berta_list, label='Berta', linewidth=2)
plt.plot(p_vals, adabi_list, label='Adabi', linewidth=2)
plt.xlabel('Werner parameter  p')
plt.ylabel('Lower bound on conditional uncertainty (bits)')
plt.title('Hierarchy of entropic uncertainty bounds for the Werner state')
plt.legend()
plt.grid(True, alpha=0.3)
import os
os.makedirs('figures/ch05', exist_ok=True)
plt.savefig('figures/ch05/uncertainty_bounds.png', dpi=300, bbox_inches='tight')
plt.show()     