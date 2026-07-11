"""
========================================================================
Title       : Measurement-Induced Entanglement Phase Transition
Book        : Quantum Entanglement, Volume 1
Chapter     : 6  --  Resource Theory and Many-Body Entanglement
Code Ref    : mipt_transition
Description : Simulates a hybrid quantum circuit of random two-qubit unitary
              gates interspersed with projective single-qubit measurements at
              rate p, and computes the half-system entanglement entropy to
              reveal the volume-law to area-law transition.
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures/ch06', exist_ok=True)
rng = np.random.default_rng(1)

def random_two_qubit_unitary():
    # Haar-random 4x4 unitary from QR of a complex Gaussian matrix, exact
    z = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph

def apply_gate(state, U, i, j, n):
    # apply a two-qubit gate U on qubits i, j of an n-qubit state vector
    state = state.reshape([2] * n)
    state = np.moveaxis(state, [i, j], [0, 1])
    shp = state.shape
    state = state.reshape(4, -1)
    state = U @ state
    state = state.reshape(shp)
    state = np.moveaxis(state, [0, 1], [i, j])
    return state.reshape(-1)

def measure_qubit(state, q, n):
    # projective Z-measurement on qubit q with Born-rule probabilities, exact
    state = state.reshape([2] * n)
    state0 = np.moveaxis(state, q, 0)[0]
    p0 = np.vdot(state0, state0).real
    outcome = 0 if rng.random() < p0 else 1
    proj = np.moveaxis(state, q, 0)
    keep = np.zeros_like(proj)
    keep[outcome] = proj[outcome]
    keep = np.moveaxis(keep, 0, q).reshape(-1)
    norm = np.linalg.norm(keep)
    return keep / norm

def half_entropy(state, n):
    # von Neumann entropy of the first n//2 qubits, exact
    a = n // 2
    M = state.reshape(2**a, 2**(n - a))
    s = np.linalg.svd(M, compute_uv=False)
    p = s**2
    p = p[p > 1e-12]
    return -np.sum(p * np.log2(p))

def run_circuit(n, p, depth, shots=30):
    vals = []
    for _ in range(shots):
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0
        for layer in range(depth):
            start = layer % 2
            for i in range(start, n - 1, 2):
                U = random_two_qubit_unitary()
                state = apply_gate(state, U, i, i + 1, n)
            for q in range(n):
                if rng.random() < p:
                    state = measure_qubit(state, q, n)
        vals.append(half_entropy(state, n))
    return np.mean(vals)

n = 8
depth = 20
p_list = list(np.linspace(0.0, 0.6, 16))

print("Measurement-induced entanglement phase transition")
print("system size n = %d, circuit depth = %d" % (n, depth))
print()
print("%-12s | %-18s" % ("meas. rate p", "half-system entropy"))
print("-" * 34)
entropies = []
for p in p_list:
    S = run_circuit(n, p, depth)
    entropies.append(S)
    print("%-12.2f | %-18.4f" % (p, S))

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(p_list, entropies, 'o-', color='darkblue', lw=2)
plt.axvline(0.16, color='red', ls='--', label='approximate critical rate')
plt.xlabel('measurement rate p')
plt.ylabel('half-system entanglement entropy')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('figures/ch06/mipt_transition.png', dpi=300, bbox_inches='tight')
plt.show()