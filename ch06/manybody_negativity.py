"""
========================================================================
Title       : Many-Body Negativity in a Spin Chain
Book        : Quantum Entanglement, Volume 1
Chapter     : 6  --  Resource Theory and Many-Body Entanglement
Code Ref    : manybody_negativity
Description : Computes the logarithmic negativity between two blocks of a spin
              chain ground state, quantifying mixed-state entanglement in a
              many-body system.
========================================================================
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# =====================================================================
#  Logarithmic negativity in the transverse-field Ising chain ground
#  state, comparing entanglement between ADJACENT vs DISTANT blocks.
#  For the mixed two-block state, entanglement entropy is NOT a valid
#  measure, but the logarithmic negativity is. Everything exact:
#  ground state, partial trace, partial transpose, trace norm.
# =====================================================================

sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))
I2 = sp.identity(2, format='csr')


def embed(op, site, N):
    out = sp.identity(1, format='csr')
    for k in range(N):
        out = sp.kron(out, op if k == site else I2, format='csr')
    return out


def ising_ring(N, J, g, h_long=1e-6):
    H = sp.csr_matrix((2**N, 2**N))
    Zs = [embed(sz, i, N) for i in range(N)]
    Xs = [embed(sx, i, N) for i in range(N)]
    for i in range(N):
        H = H - J * (Zs[i] @ Zs[(i + 1) % N])
    for i in range(N):
        H = H - g * Xs[i]
    for i in range(N):
        H = H - h_long * Zs[i]
    return H


def ground_state(N, J, g):
    H = ising_ring(N, J, g)
    vals, vecs = spla.eigsh(H, k=1, which='SA')
    return vecs[:, 0]


def reduced_two_blocks(psi, N, A_sites, B_sites):
    """Reduced density matrix of (A union B), tracing out the rest,
    from first principles."""
    rest = [i for i in range(N) if i not in A_sites and i not in B_sites]
    keep = list(A_sites) + list(B_sites)
    dimkeep = 2**len(keep)
    psi_t = psi.reshape([2] * N)
    perm = keep + rest
    psi_p = np.transpose(psi_t, perm).reshape(dimkeep, 2**len(rest))
    rho = psi_p @ psi_p.conj().T
    return rho, len(A_sites), len(B_sites)


def log_negativity(rho, dA, dB):
    """E_N = log2 || rho^{T_B} ||_1, exact (trace norm via singular values)."""
    r = rho.reshape(dA, dB, dA, dB)
    rTB = r.transpose(0, 3, 2, 1).reshape(dA * dB, dA * dB)
    sv = np.linalg.svd(rTB, compute_uv=False)
    return float(np.log2(np.sum(sv)))


N = 12
J = 1.0
# adjacent blocks: two single sites next to each other
adj_A, adj_B = [N // 2 - 1], [N // 2]
# distant blocks: two single sites far apart (opposite sides of the ring)
dist_A, dist_B = [0], [N // 2]

print(f"Transverse-field Ising ring, N = {N}")
print(" g/J    E_N(adjacent sites)   E_N(distant sites)")
for g in [0.3, 0.6, 1.0, 1.4, 1.7]:
    psi = ground_state(N, J, g)
    rho_adj, dA, dB = reduced_two_blocks(psi, N, adj_A, adj_B)
    en_adj = log_negativity(rho_adj, 2**dA, 2**dB)
    rho_dist, dA2, dB2 = reduced_two_blocks(psi, N, dist_A, dist_B)
    en_dist = log_negativity(rho_dist, 2**dA2, 2**dB2)
    print(f" {g:4.2f}    {en_adj:.4f}              {en_dist:.4f}")

g_values = np.linspace(0.05, 2.0, 40)
en_adj_scan, en_dist_scan = [], []
for g in g_values:
    psi = ground_state(N, J, g)
    ra, dA, dB = reduced_two_blocks(psi, N, adj_A, adj_B)
    en_adj_scan.append(log_negativity(ra, 2**dA, 2**dB))
    rd, dA2, dB2 = reduced_two_blocks(psi, N, dist_A, dist_B)
    en_dist_scan.append(log_negativity(rd, 2**dA2, 2**dB2))

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(g_values, en_adj_scan, marker='o', markersize=4, linewidth=2,
        label='adjacent sites')
ax.plot(g_values, en_dist_scan, marker='s', markersize=4, linewidth=2,
        label='distant sites')
ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.2,
           label='critical point g/J = 1')
ax.set_xlabel('transverse field  g / J')
ax.set_ylabel('logarithmic negativity  $E_N$')
ax.legend(loc='upper right', framealpha=1.0)
ax.grid(True, alpha=0.3)
import os
os.makedirs('figures/ch06', exist_ok=True)
plt.savefig('figures/ch06/manybody_negativity.png', dpi=200, bbox_inches='tight')
plt.show()