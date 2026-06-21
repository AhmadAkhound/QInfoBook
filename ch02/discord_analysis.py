import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def entropy(rho):
    """Calculates Von Neumann Entropy."""
    evs = np.linalg.eigvalsh(rho)
    evs = evs[evs > 1e-12]
    return -np.sum(evs * np.log2(evs))

def werner_state(p):
    """Generates 2-qubit Werner state."""
    identity = np.eye(4) / 4
    psi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_bell = np.outer(psi_plus, psi_plus)
    return p * rho_bell + (1 - p) * identity

def local_dephasing_deficit(p):
    """Calculates deficit via dephasing in computational basis."""
    rho = werner_state(p)
    rho_diag = np.diag(np.diag(rho))
    return max(0, entropy(rho_diag) - entropy(rho))

def quantum_discord(p):
    rho = werner_state(p)

    # 1. Total Correlation
    rho_a = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    rho_b = np.trace(rho.reshape(2, 2, 2, 2), axis1=0, axis2=2)
    total_corr = entropy(rho_a) + entropy(rho_b) - entropy(rho)

    # 2. Classical Correlation
    def classical_func(params):
        theta, phi = params
        v = np.array([np.cos(theta), np.exp(1j*phi)*np.sin(theta)])
        v_perp = np.array([np.sin(theta), -np.exp(1j*phi)*np.cos(theta)])
        proj = [np.outer(v, v.conj()), np.outer(v_perp, v_perp.conj())]
        
        cond_entropy = 0
        for P in proj:
            M = np.kron(np.eye(2), P)
            rho_reduced = M @ rho @ M.conj().T
            prob = np.trace(rho_reduced).real
            if prob > 1e-12:
                rho_a_cond = np.trace(rho_reduced.reshape(2, 2, 2, 2), axis1=1, axis2=3) / prob
                cond_entropy += prob * entropy(rho_a_cond)
        return cond_entropy

    res = minimize(classical_func, [0.5, 0.5], bounds=[(0, np.pi), (0, 2*np.pi)])
    max_classical_corr = entropy(rho_a) - res.fun
    discord = max(0, total_corr - max_classical_corr)
    
    # 3. Deficit
    deficit = local_dephasing_deficit(p)
    
    return total_corr, max_classical_corr, discord, deficit

# Execution
p_values = np.linspace(0, 1, 15)
results = [quantum_discord(p) for p in p_values]

# Formatting Table
print(f"{'p':<12} | {'Total':<12} | {'Classical':<12} | {'Discord':<12} | {'Deficit':<15}")
print("-" * 75)
for p, (t, c, d, defic) in zip(p_values, results):
    print(f"{p:<12.4f} | {t:<12.4f} | {c:<12.4f} | {d:<12.4f} | {defic:<15.4f}")

# Plotting
p_fine = np.linspace(0, 1, 50)
results_fine = np.array([quantum_discord(p) for p in p_fine])

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(p_fine, results_fine[:, 0], 'k-', label='Total Correlation')
plt.plot(p_fine, results_fine[:, 1], 'b--', label='Classical Correlation')
plt.plot(p_fine, results_fine[:, 2], 'r-', label='Quantum Discord')
plt.plot(p_fine, results_fine[:, 3], 'g-.', lw=2, label='Local Dephasing Deficit') # Deficit added
plt.axvline(x=1/3, color='gray', linestyle=':', label='Entanglement Boundary')
plt.xlabel('Purity parameter (p)')
plt.ylabel('Correlation Measures (Bits)')
plt.title('Correlation Measures in Werner State')
plt.legend()
plt.grid(alpha=0.3)
plt.show()