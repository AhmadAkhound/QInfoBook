import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Simulation of Rényi Entropy and its Scaling Behavior
# Analyzing the effect of order parameter alpha on Werner State
# ---------------------------------------------------------

def calculate_renyi_entropy(rho, alpha):
    """Calculates the Renyi entropy of order alpha for a density matrix rho."""
    eigvals = np.linalg.eigvalsh(rho)
    # Filter out zero or near-zero eigenvalues to avoid numerical instability
    eigvals = eigvals[eigvals > 1e-15]

    if abs(alpha - 1.0) < 1e-7:
        # Renyi alpha=1 is the von Neumann entropy
        return -np.sum(eigvals * np.log2(eigvals))
    else:
        # General formula: (1/(1-alpha)) * log2(sum(p_i^alpha))
        sum_power = np.sum(eigvals**alpha)
        return (1 / (1 - alpha)) * np.log2(sum_power)

# 1. Setup for Werner State with high purity (p = 0.9)
p = 0.9
psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
rho_pure = np.outer(psi_minus, psi_minus)
identity_4 = np.eye(4) / 4
rho_werner = p * rho_pure + (1 - p) * identity_4

# 2. Table Generation for specific alpha values
alpha_targets = [0.5, 1.0, 2.0, 5.0]
print(f"{'Alpha':<8} | {'Renyi Entropy (S_alpha)':<20}")
print("-" * 35)

for alpha in alpha_targets:
    s_alpha = calculate_renyi_entropy(rho_werner, alpha)
    print(f"{alpha:<8.1f} | {s_alpha:<20.4f}")

# 3. Visualization of the Scaling Behavior
alpha_fine = np.linspace(0.1, 10, 200)
s_values = [calculate_renyi_entropy(rho_werner, a) for a in alpha_fine]

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(alpha_fine, s_values, 'g-', linewidth=2, label=rf'Rényi Entropy ($p={p}$)')

# Adding markers for specific points
for alpha in alpha_targets:
    val = calculate_renyi_entropy(rho_werner, alpha)
    plt.plot(alpha, val, 'ro')
    plt.annotate(f'({alpha}, {val:.2f})', (alpha, val), textcoords="offset points", xytext=(0,10), ha='center')

plt.axvline(x=1.0, color='k', linestyle='--', alpha=0.3, label='von Neumann Limit')
plt.xlabel(r'Order Parameter ($\alpha$)')
plt.ylabel(r'$S_\alpha$ (Bits)')
plt.title('Scaling Behavior of Rényi Entropy in Werner States')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()