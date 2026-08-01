"""
========================================================================
Title       : Comparison of the Simon and Duan Criteria
Book        : Quantum Entanglement, Volume 1
Chapter     : 7  --  Continuous Variables and Correlations Beyond Entanglement
Code Ref    : simon_duan
Description : Compares the Simon and symmetric Duan entanglement criteria
              for the two-mode squeezed vacuum using a consistent
              quadrature convention.
========================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Convention:
#
#   [x_j, p_k] = i delta_jk
#   Var_vac(x) = Var_vac(p) = 1/2
#
# For the two-mode squeezed vacuum:
#
#   nu_tilde_min = (1/2) exp(-2r)
#
# and, with
#
#   u = x_1 - x_2,
#   v = p_1 + p_2,
#
#   Var(u) + Var(v) = 2 exp(-2r).
#
# Under this convention, every separable state satisfies
#
#   Var(u) + Var(v) >= 2.
#
# Hence both the Simon and symmetric Duan criteria detect the TMSV
# as entangled for every r > 0.
# ---------------------------------------------------------------------

os.makedirs("figures/ch07", exist_ok=True)

# Continuous squeezing range used for the figure
r = np.linspace(0.0, 1.0, 300)

# Simon criterion:
# entanglement is detected when nu_tilde_min < 1/2
nu_tilde = 0.5 * np.exp(-2.0 * r)
simon_margin = 0.5 - nu_tilde

# Symmetric Duan criterion:
# entanglement is detected when Var(u) + Var(v) < 2
duan_sum = 2.0 * np.exp(-2.0 * r)
duan_margin = 2.0 - duan_sum


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    r,
    simon_margin,
    color="navy",
    linewidth=2,
    label=r"Simon margin: $1/2-\tilde{\nu}_{\min}$",
)

ax.plot(
    r,
    duan_margin,
    color="darkred",
    linewidth=2,
    label=r"Duan margin: $2-[\mathrm{Var}(u)+\mathrm{Var}(v)]$",
)

ax.axhline(
    0.0,
    color="gray",
    linewidth=1,
)

ax.set_xlabel("squeezing parameter  r")
ax.set_ylabel("detection margin (positive = entanglement detected)")
ax.set_title("Simon and Symmetric Duan Criteria for the TMSV")
ax.legend()
ax.grid(alpha=0.3)

figure_path = "figures/ch07/simon_vs_duan.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


# ---------------------------------------------------------------------
# Numerical comparison at selected squeezing values
# ---------------------------------------------------------------------

selected_r = [0.1, 0.2, 0.3466, 0.5, 0.8]

print(
    "%-8s | %-12s | %-12s | %-10s | %-10s"
    % ("r", "nu_tilde", "Duan sum", "Simon", "Duan")
)
print("-" * 60)

for rv in selected_r:
    nt = 0.5 * np.exp(-2.0 * rv)
    ds = 2.0 * np.exp(-2.0 * rv)

    simon_result = "detected" if nt < 0.5 else "no"
    duan_result = "detected" if ds < 2.0 else "no"

    print(
        "%-8.4f | %-12.4f | %-12.4f | %-10s | %-10s"
        % (rv, nt, ds, simon_result, duan_result)
    )

print(f"\nFigure saved to: {figure_path}")