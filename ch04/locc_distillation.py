"""
========================================================================
Title       : LOCC Entanglement Distillation
Book        : Quantum Entanglement, Volume 1
Chapter     : 4  --  Operational Measures and the Cost of Entanglement
Code Ref    : locc_distillation
Description : Simulates an entanglement distillation protocol under local
              operations and classical communication, tracking the growth of
              the output fidelity across successive rounds.
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt


def shannon_entropy(lambdas):
    """Von Neumann entropy of a Schmidt spectrum (in bits)."""
    l = np.array(lambdas, dtype=float)
    l = l[l > 0]
    return -np.sum(l * np.log2(l))


def majorized_by(l_init, l_target):
    """Nielsen's theorem: |psi> -> |phi> under LOCC iff lambda_psi is
    majorized by lambda_phi. Returns True if the conversion is possible."""
    s_init = np.sort(l_init)[::-1]
    s_target = np.sort(l_target)[::-1]
    return np.all(np.cumsum(s_init) <= np.cumsum(s_target) + 1e-12)


# --- 1. Nielsen majorization test for LOCC convertibility ---
test_cases = [
    ([0.8, 0.2], [0.6, 0.4], "increase entanglement"),
    ([0.5, 0.5], [0.9, 0.1], "maximal to partial"),
    ([0.7, 0.3], [0.7, 0.3], "identity map"),
    ([1.0, 0.0], [0.5, 0.5], "create from product"),
]

print(f"{'Row':<4} | {'Initial':<12} | {'Target':<12} | {'LOCC?':<6} | Entropy change")
print("-" * 66)
for i, (init, target, desc) in enumerate(test_cases, 1):
    possible = majorized_by(init, target)
    h_i, h_t = shannon_entropy(init), shannon_entropy(target)
    status = "Yes" if possible else "No"
    print(f"{i:<4} | {str(init):<12} | {str(target):<12} | {status:<6} | {h_i:.2f} -> {h_t:.2f}")


# --- 2. BBPSSW recurrence distillation on Werner input ---
def bbpssw_step(F):
    """One BBPSSW round on two identical Werner copies of fidelity F.
    Returns (F_next, p_success). Valid recurrence for Werner/isotropic input."""
    p_succ = F**2 + (2.0/3.0)*F*(1-F) + (5.0/9.0)*(1-F)**2
    F_next = (F**2 + (1.0/9.0)*(1-F)**2) / p_succ
    return F_next, p_succ


def distillation_curve(F0, n_steps):
    """Iterate the BBPSSW recurrence to get fidelity vs. round."""
    F = F0
    fids = [F]
    for _ in range(n_steps):
        F, _ = bbpssw_step(F)
        fids.append(F)
    return np.array(fids)


n_steps = 8
rounds = np.arange(n_steps + 1)
start_fids = [0.55, 0.65, 0.80]

plt.figure(figsize=(8, 5), dpi=150)
for F0 in start_fids:
    fids = distillation_curve(F0, n_steps)
    plt.plot(rounds, fids, marker='o', linewidth=2, label=f'$F_0={F0}$')

plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.6, label='Bell state ($F=1$)')
plt.axhline(y=0.5, color='k', linestyle=':', alpha=0.7, label='Threshold ($F=1/2$)')
plt.title("BBPSSW distillation: fidelity vs. number of rounds")
plt.xlabel("Distillation round")
plt.ylabel("Fidelity to Bell state $F$")
plt.ylim(0.45, 1.02)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()
import os
os.makedirs('figures/ch04', exist_ok=True)
plt.savefig('figures/ch04/distillation_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the fidelity table for the text
print("\nBBPSSW fidelity per round (Werner input):")
print("Round | " + " | ".join(f"F0={F0}" for F0 in start_fids))
for r in rounds:
    row = [distillation_curve(F0, n_steps)[r] for F0 in start_fids]
    print(f"{r:<5} | " + " | ".join(f"{v:.4f}" for v in row))