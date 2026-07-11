"""
========================================================================
Title       : Quantum Discord and Classical Correlation in Werner States
Book        : Quantum Entanglement, Volume 1
Chapter     : 7  --  Continuous Variables and Correlations Beyond Entanglement
Code Ref    : discord_analysis
Description : Computes the total mutual information, classical correlation,
              quantum discord, and quantum deficit of the Werner state as a
              function of the mixing parameter.
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def entropy(rho):
    # von Neumann entropy in bits
    evs = np.linalg.eigvalsh(rho)
    evs = evs[evs > 1e-12]
    return -np.sum(evs * np.log2(evs))

def werner_state(p):
    # two-qubit Werner state: mixture of Bell state and maximally mixed state
    identity = np.eye(4) / 4
    psi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_bell = np.outer(psi_plus, psi_plus)
    return p * rho_bell + (1 - p) * identity

def local_dephasing_deficit(p):
    # quantum deficit via local dephasing in the computational basis
    rho = werner_state(p)
    rho_diag = np.diag(np.diag(rho))
    return max(0.0, entropy(rho_diag) - entropy(rho))

def quantum_discord(p):
    rho = werner_state(p)

    # total correlation: quantum mutual information
    rho_a = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    rho_b = np.trace(rho.reshape(2, 2, 2, 2), axis1=0, axis2=2)
    total_corr = entropy(rho_a) + entropy(rho_b) - entropy(rho)

    # classical correlation: maximized over local projective measurements on B
    def classical_func(params):
        theta, phi = params
        v = np.array([np.cos(theta), np.exp(1j * phi) * np.sin(theta)])
        v_perp = np.array([np.sin(theta), -np.exp(1j * phi) * np.cos(theta)])
        proj = [np.outer(v, v.conj()), np.outer(v_perp, v_perp.conj())]
        cond_entropy = 0.0
        for P in proj:
            M = np.kron(np.eye(2), P)
            rho_reduced = M @ rho @ M.conj().T
            prob = np.trace(rho_reduced).real
            if prob > 1e-12:
                rho_a_cond = np.trace(rho_reduced.reshape(2, 2, 2, 2), axis1=1, axis2=3) / prob
                cond_entropy += prob * entropy(rho_a_cond)
        return cond_entropy

    res = minimize(classical_func, [0.5, 0.5], bounds=[(0, np.pi), (0, 2 * np.pi)])
    max_classical_corr = entropy(rho_a) - res.fun
    discord = max(0.0, total_corr - max_classical_corr)
    deficit = local_dephasing_deficit(p)
    return total_corr, max_classical_corr, discord, deficit

p_values = np.linspace(0, 1, 15)
results = [quantum_discord(p) for p in p_values]

print("%-10s | %-10s | %-10s | %-10s | %-10s" % ("p", "Total", "Classical", "Discord", "Deficit"))
print("-" * 62)
for p, (t, c, d, defic) in zip(p_values, results):
    print("%-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f" % (p, t, c, d, defic))

p_fine = np.linspace(0, 1, 50)
results_fine = np.array([quantum_discord(p) for p in p_fine])

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(p_fine, results_fine[:, 0], "k-", label="Total correlation")
plt.plot(p_fine, results_fine[:, 1], "b--", label="Classical correlation")
plt.plot(p_fine, results_fine[:, 2], "r-", label="Quantum discord")
plt.plot(p_fine, results_fine[:, 3], "g-.", lw=2, label="Local dephasing deficit")
plt.axvline(x=1/3, color="gray", linestyle=":", label="Entanglement boundary")
plt.xlabel("Werner parameter p")
plt.ylabel("Correlation measures (bits)")
plt.legend()
plt.grid(alpha=0.3)
import os
os.makedirs("figures/ch07", exist_ok=True)
plt.savefig("figures/ch07/discord_analysis.png", dpi=300, bbox_inches="tight")
plt.show()