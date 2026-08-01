"""
========================================================================
Title       : Graph States, Stabilizers, and Entanglement Structure
Book        : Quantum Entanglement, Volume 1
Chapter     : 5  --  Uncertainty Relations, Witnesses, and Multipartite Entanglement
Code Ref    : graph_states
Description : Builds graph states from vertices and edges via controlled-Z
              gates, verifies the vertex stabilizer generators, checks the
              local equivalence of the star graph to the GHZ state, and
              computes the bipartite entanglement entropy of star and linear
              graphs from first principles.
========================================================================
"""
import numpy as np
from itertools import product

# single-qubit operators
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)


def kron_list(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def cz(n, a, b):
    # controlled-Z on qubits a, b of an n-qubit register, exact
    dim = 2**n
    diag = np.ones(dim, dtype=complex)
    for i in range(dim):
        bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[a] == 1 and bits[b] == 1:
            diag[i] = -1
    return np.diag(diag)


def graph_state(n, edges):
    # |G> = prod_{(u,v) in E} CZ_uv |+>^{⊗n}
    psi = kron_list([plus] * n)
    for (u, v) in edges:
        psi = cz(n, u, v) @ psi
    return psi


def stabilizer_generator(n, j, edges):
    # g_j = X_j ⊗ (⊗_{k in N_j} Z_k)
    neighbors = [b for (a, b) in edges if a == j] + \
                [a for (a, b) in edges if b == j]
    ops = []
    for q in range(n):
        if q == j:
            ops.append(X)
        elif q in neighbors:
            ops.append(Z)
        else:
            ops.append(I2)
    return kron_list(ops)


def entanglement_entropy(psi, n, part_A):
    # von Neumann entropy of the reduced state on subset part_A, exact
    part_B = [q for q in range(n) if q not in part_A]
    perm = part_A + part_B
    tensor = psi.reshape([2] * n).transpose(perm)
    M = tensor.reshape(2**len(part_A), 2**len(part_B))
    s = np.linalg.svd(M, compute_uv=False)
    p = s**2
    p = p[p > 1e-12]
    return -np.sum(p * np.log2(p))


# ---------- star graph on 3 qubits: center = 0 ----------
star_edges = [(0, 1), (0, 2)]
psi_star = graph_state(3, star_edges)

print("=== 3-qubit star graph ===")
print("stabilizer eigenvalues (should all be +1):")
for j in range(3):
    g = stabilizer_generator(3, j, star_edges)
    val = np.real(np.vdot(psi_star, g @ psi_star))
    print("  g_%d : <G|g|G> = %+.3f" % (j, val))

# apply Hadamard on the two peripheral qubits -> should give GHZ
Hop = kron_list([I2, H, H])
psi_transformed = Hop @ psi_star
ghz = np.zeros(8, dtype=complex)
ghz[0] = ghz[7] = 1 / np.sqrt(2)
overlap = abs(np.vdot(ghz, psi_transformed))**2
print("overlap with GHZ after local Hadamards: %.4f" % overlap)

# ---------- entanglement entropy: star vs linear (4 qubits) ----------
print()
print("=== entanglement entropy across a cut (qubit 0 | rest) ===")
star4 = [(0, 1), (0, 2), (0, 3)]
linear4 = [(0, 1), (1, 2), (2, 3)]
psi_s4 = graph_state(4, star4)
psi_l4 = graph_state(4, linear4)
print("star graph  S(0|rest)   = %.4f" % entanglement_entropy(psi_s4, 4, [0]))
print("linear graph S(0|rest)  = %.4f" % entanglement_entropy(psi_l4, 4, [0]))
print("linear graph S(01|rest) = %.4f" % entanglement_entropy(psi_l4, 4, [0, 1]))