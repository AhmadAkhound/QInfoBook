"""
========================================================================
Title       : Geometric Measure of Entanglement of the GHZ State
Book        : Quantum Entanglement, Volume 1
Chapter     : 3  --  Geometric Measures and the Partial Transpose
Code Ref    : geometric_ghz
Description : Computes the geometric measure of entanglement of the
              three-qubit GHZ state from the maximal overlap with fully
              product states.
========================================================================
"""
import numpy as np


def product_state(alpha, beta, gamma):
    """Three-qubit product state |a>|b>|c> with real single-qubit angles."""
    a = np.array([np.cos(alpha), np.sin(alpha)])
    b = np.array([np.cos(beta), np.sin(beta)])
    c = np.array([np.cos(gamma), np.sin(gamma)])
    return np.kron(np.kron(a, b), c)


# GHZ state (1/sqrt2)(|000> + |111>)
ghz = np.zeros(8)
ghz[0] = 1 / np.sqrt(2)   # |000>
ghz[7] = 1 / np.sqrt(2)   # |111>


def overlap(alpha, beta, gamma):
    """Absolute overlap |<phi|GHZ>|."""
    phi = product_state(alpha, beta, gamma)
    return abs(np.dot(phi, ghz))


# ---------------------------------------------------
# 1) Symmetric branch alpha = beta = gamma = theta
# ---------------------------------------------------

print("Symmetric branch (alpha=beta=gamma=theta):")
print("theta(deg)   overlap")
for theta_deg in [0, 22.5, 45, 67.5, 90]:
    t = np.deg2rad(theta_deg)
    print(f"{theta_deg:<10}   {overlap(t, t, t):.4f}")
print()


# ---------------------------------------------------
# 2) Global search over independent angles
# ---------------------------------------------------

best = 0.0
best_angles = None

grid = np.linspace(0, np.pi, 73)   # 2.5 degree steps

for a in grid:
    for b in grid:
        for c in grid:
            ov = overlap(a, b, c)
            if ov > best:
                best = ov
                best_angles = (a, b, c)

lambda_max = best
e_geometric = 1 - lambda_max ** 2

print("Global maximum over independent angles:")
print("Max overlap (Lambda_max):", round(lambda_max, 4))
print("Best angles (deg):",
      tuple(round(np.rad2deg(x), 1) for x in best_angles))
print("Geometric measure E_G = 1 - Lambda_max^2:", round(e_geometric, 4))