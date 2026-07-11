"""
========================================================================
Title       : Hidden Entanglement and Procrustean Filtering
Book        : Quantum Entanglement, Volume 1
Chapter     : 4  --  Operational Measures and the Cost of Entanglement
Code Ref    : hidden_entanglement
Description : Computes the concurrence of an asymmetric mixed two-qubit state
              before and after local Procrustean filtering, locating the
              crossover where filtering starts to recover entanglement.
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def concurrence(rho):
    # spin-flip operator sigma_y tensor sigma_y
    sy = np.array([[0, -1j], [1j, 0]])
    syy = np.kron(sy, sy)
    rho_tilde = syy @ rho.conj() @ syy
    R = rho @ rho_tilde
    ev = np.linalg.eigvals(R)
    ev = np.sqrt(np.maximum(ev.real, 0))
    ev = np.sort(ev)[::-1]
    return max(0.0, ev[0] - ev[1] - ev[2] - ev[3])

def mixed_state(p, alpha, beta):
    psi = np.array([alpha, 0, 0, beta])
    rho_pure = np.outer(psi, psi.conj())
    return p * rho_pure + (1 - p) * np.eye(4) / 4

def apply_filter(rho, f1, f2):
    A = np.array([[f1, 0], [0, f2]])
    F = np.kron(A, np.eye(2))
    rho_f = F @ rho @ F.conj().T
    return rho_f / np.trace(rho_f).real

alpha = 0.9
beta = np.sqrt(1 - alpha**2)
f1, f2 = 0.484, 1.0

p_values = np.linspace(0.4, 0.9, 7)
print("%-10s | %-14s | %-14s" % ("p", "C_initial", "C_filtered"))
print("-" * 44)
rows = []
for p in p_values:
    rho = mixed_state(p, alpha, beta)
    c_init = concurrence(rho)
    c_filt = concurrence(apply_filter(rho, f1, f2))
    rows.append((p, c_init, c_filt))
    print("%-10.4f | %-14.4f | %-14.4f" % (p, c_init, c_filt))

p_fine = np.linspace(0.3, 0.95, 60)
c_init_fine = [concurrence(mixed_state(p, alpha, beta)) for p in p_fine]
c_filt_fine = [concurrence(apply_filter(mixed_state(p, alpha, beta), f1, f2)) for p in p_fine]

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(p_fine, c_init_fine, "b-", lw=2, label="initial concurrence")
plt.plot(p_fine, c_filt_fine, "r--", lw=2, label="filtered concurrence")
plt.xlabel("mixing parameter p")
plt.ylabel("concurrence")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/ch04/hidden_entanglement.png", dpi=300, bbox_inches="tight")
plt.show()