"""
========================================================================
Title       : Comparison of Entanglement Measures for Pure States
Book        : Quantum Entanglement, Volume 1
Chapter     : 3  --  entanglement Measures and the Partial Transpose
Code Ref    : measures_comparison
Description : Compares the geometric, Bures, and negativity measures of a
              pure two-qubit state as functions of the concurrence, showing
              their monotonic relationships.
========================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def geometric_measure(C):
    """Geometric measure of entanglement for two qubits as a function of concurrence."""
    return 0.5 * (1 - np.sqrt(1 - C ** 2))


def bures_measure(C):
    """Bures measure of entanglement for two qubits as a function of concurrence."""
    return 2 - 2 * np.sqrt((1 + np.sqrt(1 - C ** 2)) / 2)


def negativity(C):
    """Negativity of a pure two-qubit state equals C/2."""
    return C / 2


# ---------------------------------------------------
# Numerical table over concurrence
# ---------------------------------------------------

C_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

rows = []

for C in C_values:
    rows.append([C, geometric_measure(C), bures_measure(C), negativity(C)])

columns = ["Concurrence C", "E_G", "E_B", "Negativity"]

df = pd.DataFrame(rows, columns=columns)

print(df.to_string(index=False))


# ---------------------------------------------------
# Figure: three measures versus concurrence
# ---------------------------------------------------

C = np.linspace(0, 1, 300)

plt.figure(figsize=(8, 5), dpi=300)

plt.plot(C, geometric_measure(C), linewidth=2, label=r"Geometric measure $E_G$")
plt.plot(C, bures_measure(C), linewidth=2, linestyle="--", label=r"Bures measure $E_B$")
plt.plot(C, negativity(C), linewidth=2, linestyle=":", label=r"Negativity $\mathcal{N}$")

plt.xlabel("Concurrence $C$")
plt.ylabel("Entanglement measure")
plt.title("Three geometric measures as functions of concurrence")
plt.legend()
plt.grid(alpha=0.3)
import os
os.makedirs('figures/ch03', exist_ok=True)
plt.savefig('figures/ch03/measures_comparison.png', dpi=300, bbox_inches='tight')
plt.show()