"""
========================================================================
Title       : Concurrence and Entanglement of Formation of the Werner State
Book        : Quantum Entanglement, Volume 1
Chapter     : 2  --  Entropic Foundations and Structural Measures
Code Ref    : concurrence_eof
Description : Computes the closed-form concurrence of the Werner state and the
              entanglement of formation through the Hill-Wootters relation as
              functions of the mixing parameter.
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def concurrence_werner(p):
    # Closed-form concurrence of the Werner state
    return max(0, (3 * p - 1) / 2)

def binary_entropy(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def eof_from_concurrence(C):
    # Entanglement of formation via the Hill-Wootters relation
    if C <= 0:
        return 0.0
    arg = (1 + np.sqrt(1 - C**2)) / 2
    return binary_entropy(arg)

# Range of mixing parameter
p_values = np.linspace(0, 1, 300)
C_values = [concurrence_werner(p) for p in p_values]
E_values = [eof_from_concurrence(concurrence_werner(p)) for p in p_values]

# Print key checkpoints for verification
print(f"{'p':<8} | {'Concurrence':<12} | {'EoF':<10}")
print("-" * 36)
for p in [0.2, 1/3, 0.5, 0.8, 1.0]:
    C = concurrence_werner(p)
    E = eof_from_concurrence(C)
    print(f"{p:<8.3f} | {C:<12.4f} | {E:<10.4f}")

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(p_values, C_values, 'b-', linewidth=2, label='Concurrence C')
plt.plot(p_values, E_values, 'r--', linewidth=2, label='Entanglement of Formation')
plt.axvline(x=1/3, color='gray', linestyle=':', linewidth=1.5,
            label='Separability threshold p = 1/3')
plt.xlabel('Mixing parameter p')
plt.ylabel('Entanglement measure')
plt.title('Concurrence and Entanglement of Formation of the Werner State')
plt.legend()
plt.grid(True, alpha=0.2)
import os
os.makedirs('figures/ch02', exist_ok=True)
plt.savefig('figures/ch02/concurrence_eof_plot.png', dpi=300, bbox_inches='tight')
plt.show()