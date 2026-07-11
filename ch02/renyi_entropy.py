"""
========================================================================
Title       : Renyi Entropy and Its von Neumann Limit
Book        : Quantum Entanglement, Volume 1
Chapter     : 2  --  Entropic Foundations and Structural Measures
Code Ref    : renyi_entropy
Description : Computes the Renyi entropy of a high-purity Werner state as a
              function of the order parameter alpha and shows its convergence
              to the von Neumann entropy in the limit alpha -> 1.
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
def calculate_renyi_entropy(rho, alpha):
    # Renyi entropy of order alpha for a density matrix rho
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    if abs(alpha - 1.0) < 1e-7:
        # Order 1 reduces to the von Neumann entropy
        return -np.sum(eigvals * np.log2(eigvals))
    else:
        sum_power = np.sum(eigvals**alpha)
        return (1 / (1 - alpha)) * np.log2(sum_power)
# Werner state with high purity (p = 0.9)
p = 0.9
phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
rho_pure = np.outer(phi_plus, phi_plus)
identity_4 = np.eye(4) / 4
rho_werner = p * rho_pure + (1 - p) * identity_4
# Numerical table
alpha_targets = [0.5, 1.0, 2.0, 5.0]
print(f"{'Alpha':<8} | {'Renyi Entropy (S_alpha)':<20}")
print("-" * 35)
for alpha in alpha_targets:
    s_alpha = calculate_renyi_entropy(rho_werner, alpha)
    print(f"{alpha:<8.1f} | {s_alpha:<20.4f}")
# Scaling behavior plot
alpha_fine = np.linspace(0.1, 10, 200)
s_values = [calculate_renyi_entropy(rho_werner, a) for a in alpha_fine]
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(alpha_fine, s_values, 'g-', linewidth=2, label=rf'Renyi Entropy ($p={p}$)')
for alpha in alpha_targets:
    val = calculate_renyi_entropy(rho_werner, alpha)
    plt.plot(alpha, val, 'ro')
    plt.annotate(f'({alpha}, {val:.2f})', (alpha, val),
                textcoords="offset points", xytext=(0,10), ha='center')
plt.axvline(x=1.0, color='k', linestyle='--', alpha=0.3, label='von Neumann Limit')
plt.xlabel(r'Order Parameter ($\alpha$)')
plt.ylabel(r'$S_\alpha$ (Bits)')
plt.title('Scaling Behavior of Renyi Entropy in Werner States')
plt.legend()
plt.grid(True, alpha=0.2)
import os
os.makedirs('figures/ch02', exist_ok=True)
plt.savefig('figures/ch02/renyi_scaling.png', dpi=300, bbox_inches='tight')
plt.show()