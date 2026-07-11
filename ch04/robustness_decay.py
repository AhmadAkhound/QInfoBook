"""
========================================================================
Title       : Robustness of Entanglement under Noise
Book        : Quantum Entanglement, Volume 1
Chapter     : 4  --  Operational Measures and the Cost of Entanglement
Code Ref    : robustness_decay
Description : Concurrence and random robustness R_r = 2C of the Werner state
              rho(p) = p|Phi+><Phi+| + (1-p) I/4, where p is the entanglement
              parameter (p=1 pure Bell state, p=1/3 separability boundary).
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt


def concurrence(rho):
    """Concurrence of a 2-qubit density matrix (Wootters formula)."""
    sigma_y = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sigma_y, sigma_y)
    rho_tilde = yy @ rho.conj() @ yy
    R = rho @ rho_tilde
    evs = np.sort(np.real(np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)), 0))))[::-1]
    return max(0.0, evs[0] - evs[1] - evs[2] - evs[3])


def werner_robustness(p_steps=20):
    """Concurrence and random robustness R_r = 2C of the Werner state
    rho(p) = p|psi-><psi-| + (1-p) I/4, where p is the entanglement
    parameter (p=1 pure singlet, p=1/3 separability boundary)."""
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)   # phi_plus |Phi+>
    rho_pure = np.outer(psi, psi)
    noise = np.eye(4) / 4

    p_values = np.linspace(0, 1, p_steps)
    conc_values = []
    rob_values = []
    rob_analytic = []
    for p in p_values:
        rho = p * rho_pure + (1 - p) * noise
        c = concurrence(rho)
        conc_values.append(c)
        rob_values.append(2 * c)                  # R_r = 2C, numerical
        rob_analytic.append(max(0.0, 3 * p - 1))  # closed form
    return (p_values, np.array(conc_values),
            np.array(rob_values), np.array(rob_analytic))


p_vals, c_vals, r_vals, r_exact = werner_robustness()

print(f"{'p (ent.)':<10} | {'Concurrence':<12} | {'R_r=2C':<10} | {'R_r exact':<10}")
print("-" * 50)
for p, c, r, re in zip(p_vals, c_vals, r_vals, r_exact):
    print(f"{p:<10.4f} | {c:<12.4f} | {r:<10.4f} | {re:<10.4f}")

p_c = 1.0 / 3.0
print(f"\nCritical entanglement parameter (entanglement vanishes): p_c = {p_c:.4f}")
print(f"Max robustness at p=1: R_r = {r_vals[-1]:.4f}")

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(p_vals, c_vals, 'b-o', lw=2, markersize=4, label='Concurrence $C$')
plt.plot(p_vals, r_vals, 'm-s', lw=2, markersize=4, label='Robustness $R_r=2C$')
plt.axvline(x=p_c, color='red', linestyle='--', alpha=0.7,
            label=r'Boundary $p_c=1/3$')
plt.axhline(y=0, color='k', lw=0.8)
plt.xlabel('Entanglement parameter $p$')
plt.ylabel('Entanglement measure')
plt.title('Concurrence and robustness of the Werner state')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
import os
os.makedirs('figures/ch04', exist_ok=True)
plt.savefig('figures/ch04/robustness_decay.png', dpi=300, bbox_inches='tight')
plt.show()