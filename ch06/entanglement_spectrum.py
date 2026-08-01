"""
========================================================================
Title       : Entanglement Spectrum of a Spin Chain
Book        : Quantum Entanglement, Volume 1
Chapter     : 6  --  Resource Theory and Many-Body Entanglement
Code Ref    : entanglement_spectrum
Description : Computes the entanglement spectrum from the eigenvalues of the
              reduced density matrix of a spin chain ground state, revealing
              its low-lying structure.
========================================================================
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# =====================================================================
#  Entanglement spectrum of the transverse-field Ising chain ground
#  state. To display the near-degeneracy of the low-lying entanglement
#  levels in the ordered regime, we do not add a longitudinal field.
#  Instead, among the numerically obtained low-energy eigenvectors, we
#  select a state in the Z2-even parity sector. The calculation is an
#  exact diagonalization of the finite N-site Hamiltonian up to numerical
#  eigensolver precision.
# =====================================================================

sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))
I2 = sp.identity(2, format='csr')


def embed(op, site, N):
    out = sp.identity(1, format='csr')
    for k in range(N):
        out = sp.kron(out, op if k == site else I2, format='csr')
    return out


def ising_ring(N, J, g):
    H = sp.csr_matrix((2**N, 2**N))
    Zs = [embed(sz, i, N) for i in range(N)]
    Xs = [embed(sx, i, N) for i in range(N)]
    for i in range(N):
        H = H - J * (Zs[i] @ Zs[(i + 1) % N])
    for i in range(N):
        H = H - g * Xs[i]
    return H


def parity_operator(N):
    P = embed(sx, 0, N)
    for i in range(1, N):
        P = P @ embed(sx, i, N)
    return P


def sym_ground_state(N, J, g):
    """Exact ground state in the Z2-even sector (parity P = +1)."""
    H = ising_ring(N, J, g)
    # take the lowest few eigenvectors, then pick the parity-even one
    vals, vecs = spla.eigsh(H, k=4, which='SA')
    order = np.argsort(vals)
    vecs = vecs[:, order]
    P = parity_operator(N)
    for j in range(vecs.shape[1]):
        v = vecs[:, j]
        pv = np.vdot(v, P @ v).real                 # <v|P|v>
        if pv > 0.5:                     # parity-even
            return v
    return vecs[:, 0]


def reduced_eigs(N, J, g, tol=1e-14):
    gs = sym_ground_state(N, J, g)
    nA = N // 2
    psi = gs.reshape(2**nA, 2**(N - nA))
    rhoA = psi @ psi.conj().T
    w = np.linalg.eigvalsh(rhoA)
    w = np.sort(w)[::-1]
    return w[w > tol]


N = 10
J = 1.0
for g in [0.3, 1.0, 1.7]:
    w = reduced_eigs(N, J, g)
    xi = -np.log(w)
    print(f"\ng/J = {g}")
    print("  largest reduced eigenvalues:", np.round(w[:6], 4))
    print("  entanglement levels xi:", np.round(xi[:6], 4))
    print("  entanglement gap (xi_1 - xi_0):", round(xi[1] - xi[0], 4))

fig, ax = plt.subplots(figsize=(6, 4.5))
for g, color in [(0.3, 'C0'), (1.0, 'C1'), (1.7, 'C2')]:
    w = reduced_eigs(N, J, g)
    xi = -np.log(w)[:12]
    ax.scatter([g] * len(xi), xi, s=40, color=color, label=f'g/J = {g}')
ax.set_xlabel('transverse field  g / J')
ax.set_ylabel(r'entanglement levels  $\xi_i = -\ln \lambda_i$')
ax.legend(loc='upper left', framealpha=1.0)
ax.grid(True, alpha=0.3)
import os
os.makedirs('figures/ch06', exist_ok=True)
plt.savefig('figures/ch06/entanglement_spectrum.png', dpi=200, bbox_inches='tight')
plt.show()