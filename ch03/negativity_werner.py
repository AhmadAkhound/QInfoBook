"""
========================================================================
Title       : Negativity and Partial-Transpose Eigenvalues of the Werner State
Book        : Quantum Entanglement, Volume 1
Chapter     : 3  --  Geometric Measures and the Partial Transpose
Code Ref    : negativity_werner
Description : Computes the partial transpose of the Werner state, its negativity
              and trace distance, and tracks the critical eigenvalue that
              changes sign at the separability threshold p = 1/3.
========================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def partial_transpose(rho):
    """Partial transpose with respect to subsystem B."""
    return rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)


def trace_distance(rho, sigma):
    """Trace distance D(rho,sigma)=1/2 ||rho-sigma||_1 (valid for Hermitian args)."""
    delta = rho - sigma
    eigvals = np.linalg.eigvalsh(delta)
    return 0.5 * np.sum(np.abs(eigvals))


def negativity_from_pt(rho_pt):
    """Negativity as the sum of absolute values of the negative eigenvalues."""
    eigvals = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigvals[eigvals < 0]))


def analyze_werner(p_values):

    results = []

    # Bell state |Phi+>
    bell = np.array([1, 0, 0, 1], dtype=float) / np.sqrt(2)
    rho_bell = np.outer(bell, bell)

    identity = np.eye(4)
    maximally_mixed = identity / 4

    for p in p_values:

        rho = p * rho_bell + (1 - p) * maximally_mixed
        rho_pt = partial_transpose(rho)
        pt_eigs = np.linalg.eigvalsh(rho_pt)
        min_ev = np.min(pt_eigs)
        negativity = negativity_from_pt(rho_pt)
        td = trace_distance(rho, maximally_mixed)
        status = "Entangled (NPT)" if min_ev < -1e-12 else "Separable (PPT)"
        results.append([p, min_ev, negativity, td, status])

    columns = ["p", "Min Eigenvalue", "Negativity", "Trace Distance", "Status"]
    return pd.DataFrame(results, columns=columns)


# ---------------------------------------------------
# Generate numerical results
# ---------------------------------------------------

p_range = [0.0, 0.2, 1/3, 0.4, 0.6, 0.8, 1.0]
output_df = analyze_werner(p_range)
print(output_df.to_string(index=False))


# ---------------------------------------------------
# Eigenvalues of the partial transpose vs. p
# ---------------------------------------------------

p = np.linspace(0, 1, 300)

bell = np.array([1, 0, 0, 1], dtype=float) / np.sqrt(2)
rho_bell = np.outer(bell, bell)

ev_sorted = np.zeros((len(p), 4))
for i, x in enumerate(p):
    rho = x * rho_bell + (1 - x) * np.eye(4) / 4
    rho_pt = partial_transpose(rho)
    ev_sorted[i] = np.sort(np.linalg.eigvalsh(rho_pt))

plt.figure(figsize=(8, 5), dpi=300)

# Three degenerate positive eigenvalues (1+p)/4 and one critical (1-3p)/4
plt.plot(p, ev_sorted[:, 3], linewidth=2,
         label=r'Triple eigenvalue $(1+p)/4$')
plt.plot(p, ev_sorted[:, 0], linewidth=2, color='crimson',
         label=r'Critical eigenvalue $(1-3p)/4$')

plt.axhline(y=0, color='k', linewidth=0.8)
plt.axvline(x=1/3, linestyle='--', color='gray',
            label='Sign change at $p=1/3$')
plt.fill_between(p, ev_sorted[:, 0], 0,
                 where=(ev_sorted[:, 0] < 0), alpha=0.15, color='crimson')

plt.xlabel('Mixing parameter $p$')
plt.ylabel('Eigenvalues of $\\rho_W^{T_B}$')
plt.title('Partial-transpose eigenvalues of the Werner state')
plt.legend()
plt.grid(alpha=0.3)
import os
os.makedirs('figures/ch03', exist_ok=True)
plt.savefig('figures/ch03/negativity_werner.png', dpi=300, bbox_inches='tight')
plt.show()