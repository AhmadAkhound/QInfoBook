"""
========================================================================
Title       : Comparison of the Simon and Duan Criteria
Book        : Quantum Entanglement, Volume 1
Chapter     : 7  --  Continuous Variables and Correlations Beyond Entanglement
Code Ref    : simon_duan
Description : Compares the detection margins of the Simon and the simple Duan
              criteria for the two-mode squeezed vacuum, showing the region
              where the simple Duan criterion fails to detect entanglement.
========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures/ch07", exist_ok=True)

# convention: vacuum covariance = (1/2) I, uncertainty bound nu >= 1/2
# TMSV smallest symplectic eigenvalue of partial transpose: nu_tilde = (1/2) exp(-2r)
# simple Duan sum of variances: 2 exp(-2r), separable bound = 1

r = np.linspace(0.0, 1.0, 300)

# Simon: entangled when nu_tilde < 1/2  ->  detection margin (1/2 - nu_tilde)
nu_tilde = 0.5 * np.exp(-2 * r)
simon_margin = 0.5 - nu_tilde

# simple Duan: entangled when sum < 1  ->  detection margin (1 - sum)
duan_sum = 2 * np.exp(-2 * r)
duan_margin = 1.0 - duan_sum

# threshold where simple Duan starts to detect
r_threshold = np.log(2) / 2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r, simon_margin, color="navy", lw=2, label="Simon margin (1/2 - nu_tilde)")
ax.plot(r, duan_margin, color="darkred", lw=2, label="simple Duan margin (1 - sum)")
ax.axhline(0.0, color="gray", lw=1)
ax.axvline(r_threshold, color="green", ls="--", lw=1.5,
           label="Duan threshold r = ln2/2")
ax.fill_between(r, duan_margin, 0, where=(duan_margin < 0),
                color="orange", alpha=0.3, label="Duan blind region")
ax.set_xlabel("squeezing parameter r")
ax.set_ylabel("detection margin (positive = entanglement detected)")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig("figures/ch07/simon_vs_duan.png", dpi=300, bbox_inches="tight")
plt.show()

# print numeric comparison at selected points
print("%-8s | %-12s | %-12s | %-10s | %-10s" %
      ("r", "nu_tilde", "Duan sum", "Simon", "Duan"))
print("-" * 60)
for rv in [0.1, 0.2, 0.3466, 0.5, 0.8]:
    nt = 0.5 * np.exp(-2 * rv)
    ds = 2 * np.exp(-2 * rv)
    simon = "detected" if nt < 0.5 else "no"
    duan = "detected" if ds < 1.0 else "no"
    print("%-8.4f | %-12.4f | %-12.4f | %-10s | %-10s" % (rv, nt, ds, simon, duan))