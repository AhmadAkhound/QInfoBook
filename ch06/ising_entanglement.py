"""
========================================================================
Title       : Entanglement across the Transverse-Field Ising Transition
Book        : Quantum Entanglement, Volume 1
Chapter     : 6  --  Resource Theory and Many-Body Entanglement
Code Ref    : ising_entanglement
Description : Computes the ground-state entanglement entropy of the
              transverse-field Ising chain and its peak near the quantum
              critical point.
========================================================================
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# =====================================================================
#  Transverse-field Ising ring:  H = -J sum_i Z_i Z_{i+1} - g sum_i X_i
#  (periodic boundary, J = 1). Ground-state half-chain entanglement
#  entropy by EXACT diagonalization. An infinitesimal longitudinal field
#  is added only to lift the ground-state degeneracy and select one
#  symmetry-broken ground state near g = 0. Exact sparse Hamiltonian,
#  exact ground state, exact partial trace, and exact von Neumann
#  entropy from the reduced density-matrix eigenvalues.
# =====================================================================

sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))
I2 = sp.identity(2, format='csr')


def embed(op, site, N):
    """Embed a single-site operator at `site` into the N-spin chain."""
    out = sp.identity(1, format='csr')
    for k in range(N):
        out = sp.kron(out, op if k == site else I2, format='csr')
    return out


def ising_ring(N, J, g, h_long=0.0):
    """Sparse TFIM Hamiltonian on a ring. h_long is a tiny longitudinal
    field that selects the symmetric ground state at g=0 (physical
    infinitesimal symmetry breaking, not numerical tinkering)."""
    H = sp.csr_matrix((2**N, 2**N))
    Zs = [embed(sz, i, N) for i in range(N)]
    Xs = [embed(sx, i, N) for i in range(N)]
    for i in range(N):
        H = H - J * (Zs[i] @ Zs[(i + 1) % N])     # periodic ZZ coupling
    for i in range(N):
        H = H - g * Xs[i]                          # transverse field
    if h_long != 0.0:
        for i in range(N):
            H = H - h_long * Zs[i]                 # symmetry-breaking seed
    return H


def half_chain_entropy(N, J, g, h_long=1e-6, tol=1e-12):
    H = ising_ring(N, J, g, h_long)
    # smallest algebraic eigenvalue -> ground state, exact
    vals, vecs = spla.eigsh(H, k=1, which='SA')
    gs = vecs[:, 0]
    nA = N // 2
    psi = gs.reshape(2**nA, 2**(N - nA))
    rhoA = psi @ psi.conj().T
    w = np.linalg.eigvalsh(rhoA)
    w = w[w > tol]
    return float(-np.sum(w * np.log2(w)))


# ---- scan the transverse field for two chain lengths ----
J = 1.0
g_values = np.linspace(0.05, 2.0, 40)

results = {}
for N in [8, 12]:
    S = np.array([half_chain_entropy(N, J, g) for g in g_values])
    results[N] = S
    print(f"\nIsing ring, N = {N}, J = {J}")
    print("  g/J     S(half chain) [bits]")
    for g in [0.05, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        print(f"  {g:4.2f}    {half_chain_entropy(N, J, g):.4f}")
    gp = g_values[np.argmax(S)]
    print(f"  peak at g/J = {gp:.3f}, S = {S.max():.4f} bits")

# ---- figure: entropy vs field, two sizes, peak near critical point ----
fig, ax = plt.subplots(figsize=(6, 4.5))
for N, marker in [(8, 's'), (12, 'o')]:
    ax.plot(g_values, results[N], marker=marker, markersize=4,
            linewidth=2, label=f'N = {N}')
ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.2,
           label='critical point  g/J = 1')
ax.set_xlabel('transverse field  g / J')
ax.set_ylabel('half-chain entanglement entropy  S  [bits]')
ax.legend(loc='upper right', framealpha=1.0)
ax.grid(True, alpha=0.3)
import os
os.makedirs('figures/ch06', exist_ok=True)
plt.savefig('figures/ch06/ising_entanglement.png', dpi=200, bbox_inches='tight')
plt.show()