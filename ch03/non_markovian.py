"""
========================================================================
Title       : Markovian versus Non-Markovian Entanglement Dynamics
Book        : Quantum Entanglement, Volume 1
Chapter     : 3  --  Geometric Measures and the Partial Transpose
Code Ref    : non_markovian_revival
Description : Compares the concurrence dynamics of two qubits under a
              Markovian amplitude-damping channel and a non-Markovian
              environment, showing entanglement sudden death and revival.
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures/ch03', exist_ok=True)

def markovian_concurrence(t, gamma):
    # Markovian amplitude survival: pure exponential, no revival
    q = np.exp(-gamma * t / 2)
    return q**2

def non_markovian_concurrence(t, gamma, Omega):
    # non-Markovian amplitude survival with memory-induced oscillation
    q = np.exp(-gamma * t / 2) * (np.cos(Omega * t / 2)
                                  + (gamma / Omega) * np.sin(Omega * t / 2))
    return q**2

gamma = 1.0
Omega = 6.0            # strong coupling -> non-Markovian regime
t = np.linspace(0, 8, 400)

c_mark = markovian_concurrence(t, gamma)
c_nonmark = non_markovian_concurrence(t, gamma, Omega)

# locate the first revival: a zero of concurrence followed by an increase
death_times = []
for i in range(1, len(t) - 1):
    if c_nonmark[i] < 1e-4 and c_nonmark[i + 1] > c_nonmark[i]:
        death_times.append(t[i])

print("Markovian vs non-Markovian entanglement dynamics")
print("gamma = %.2f, Omega = %.2f" % (gamma, Omega))
print()
print("%-8s | %-14s | %-16s" % ("t", "C_markovian", "C_nonmarkovian"))
print("-" * 44)
for tv in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
    cm = markovian_concurrence(tv, gamma)
    cn = non_markovian_concurrence(tv, gamma, Omega)
    print("%-8.2f | %-14.4f | %-16.4f" % (tv, cm, cn))

if death_times:
    print()
    print("first entanglement death near t = %.3f, revival follows" % death_times[0])

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(t, c_mark, 'b-', lw=2, label='Markovian (no revival)')
plt.plot(t, c_nonmark, 'r-', lw=2, label='non-Markovian (revival)')
plt.axhline(0, color='gray', lw=0.8)
plt.xlabel('time (gamma t)')
plt.ylabel('concurrence')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('figures/ch03/non_markovian_revival.png', dpi=300, bbox_inches='tight')
plt.show()