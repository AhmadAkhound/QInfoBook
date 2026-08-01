"""
========================================================================
Title       : Hierarchy of Entanglement Measures for the Werner State
Book        : Quantum Entanglement, Volume 1
Chapter     : 4  --  Operational Measures and the Cost of Entanglement
Code Ref    : entanglement_hierarchy
Description : Computes concurrence, negativity, logarithmic negativity,
              robustness, and entanglement of formation of the Werner state
              together, comparing their growth above the p = 1/3 threshold.
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt


def binary_entropy(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def werner_measures(p):
    """All entanglement measures for the Werner state
    rho(p) = p|Phi+><Phi+| + (1-p) I/4, p = entanglement parameter
    (p=1 pure Bell state, p=1/3 separability boundary)."""
    phi = np.array([1, 0, 0, 1]) / np.sqrt(2)   # |Phi+>
    rho = p * np.outer(phi, phi) + (1 - p) * np.eye(4) / 4

    sy = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sy, sy)
    R = rho @ (yy @ rho.conj() @ yy)
    ev = np.sort(np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)), 0)))[::-1]
    C = max(0.0, ev[0] - ev[1] - ev[2] - ev[3])

    tau = C ** 2
    x = (1 + np.sqrt(1 - C ** 2)) / 2
    EoF = binary_entropy(x)

    rhoT = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    evN = np.linalg.eigvalsh(rhoT)
    Neg = sum(abs(e) for e in evN if e < 0)

    Rr = 2 * C

    # Logarithmic negativity E_N = log2(||rho^{T_B}||_1) = log2(2N+1)
    log_neg = np.log2(2 * Neg + 1)

    return C, tau, EoF, Neg, Rr, log_neg


p_values = np.linspace(0, 1, 21)
print(f"{'p':<6} | {'C':<8} | {'tau':<8} | {'EoF':<8} | {'Neg':<8} | {'R_r':<8} | {'E_N':<8}")
print("-" * 66)
data = []
for p in p_values:
    C, tau, EoF, Neg, Rr, log_neg = werner_measures(p)
    data.append((p, C, tau, EoF, Neg, Rr, log_neg))
    print(f"{p:<6.3f} | {C:<8.4f} | {tau:<8.4f} | {EoF:<8.4f} | {Neg:<8.4f} | {Rr:<8.4f} | {log_neg:<8.4f}")

print(f"\nSeparability boundary: p_c = {1/3:.4f}")
data = np.array(data)
p_c = 1.0 / 3.0

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(data[:, 0], data[:, 1], 'b-o', ms=3, label='Concurrence $C$')
plt.plot(data[:, 0], data[:, 3], 'g-^', ms=3, label='EoF $E_F$')
plt.plot(data[:, 0], data[:, 4], 'm-s', ms=3, label='Negativity $N$')
plt.plot(data[:, 0], data[:, 2], 'c-d', ms=3, label=r'Tangle $\tau$')
plt.plot(data[:, 0], data[:, 5], 'r-v', ms=3, label='Robustness $R_r$')
plt.plot(data[:, 0], data[:, 6], color='orange', marker='*', ms=4, label='Log-negativity $E_N$')
plt.axvline(x=p_c, color='red', linestyle='--', alpha=0.7, label=r'$p_c=1/3$')
plt.xlabel('Entanglement parameter $p$')
plt.ylabel('Entanglement measure')
plt.title('Hierarchy of entanglement measures for the Werner state')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
import os
os.makedirs('figures/ch04', exist_ok=True)
plt.savefig('figures/ch04/entanglement_hierarchy.png', dpi=300, bbox_inches='tight')
plt.show()