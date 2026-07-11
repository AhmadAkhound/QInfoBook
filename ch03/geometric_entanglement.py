"""
========================================================================
Title       : Geometric Measure of Entanglement
Book        : Quantum Entanglement, Volume 1
Chapter     : 3  --  Geometric Measures and the Partial Transpose
Code Ref    : geometric_entanglement
Description : Computes the geometric measure of entanglement of bipartite
              pure states from the maximal overlap with product states and
              relates it to the Schmidt coefficients.
========================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def geometric_measure_pure(coeff_matrix):
    """
    Geometric measure of entanglement for a bipartite pure state.

    The state is given by its coefficient matrix C, where
    |psi> = sum_ij C_ij |i>|j>. The singular values of C are the
    square roots of the Schmidt coefficients, so the maximal overlap
    with a product state equals the largest singular value, and
    E_G = 1 - (sigma_max)^2 = 1 - lambda_max.
    """
    # Normalize the state
    norm = np.sqrt(np.sum(np.abs(coeff_matrix) ** 2))
    C = coeff_matrix / norm

    # Singular values give sqrt(Schmidt coefficients)
    singular_values = np.linalg.svd(C, compute_uv=False)

    lambda_max = np.max(singular_values ** 2)
    max_overlap = np.sqrt(lambda_max)
    e_geometric = 1 - lambda_max

    return max_overlap, lambda_max, e_geometric


def state_matrix(theta):
    """Coefficient matrix of |psi> = cos(theta)|00> + sin(theta)|11>."""
    return np.array([[np.cos(theta), 0.0],
                     [0.0, np.sin(theta)]])


# ---------------------------------------------------
# Numerical table over the angle theta
# ---------------------------------------------------

theta_values_deg = [0.0, 22.5, 30.0, 45.0, 60.0, 67.5, 90.0]

results = []

for theta_deg in theta_values_deg:

    theta = np.deg2rad(theta_deg)

    C = state_matrix(theta)

    max_overlap, lambda_max, e_g = geometric_measure_pure(C)

    results.append([theta_deg, max_overlap, lambda_max, e_g])

columns = ["theta (deg)", "Max Overlap", "Largest Schmidt", "E_G"]

output_df = pd.DataFrame(results, columns=columns)

print(output_df.to_string(index=False))


# ---------------------------------------------------
# Continuous curve of E_G versus theta
# ---------------------------------------------------

theta = np.linspace(0, np.pi / 2, 300)

e_g_curve = []

for t in theta:

    C = state_matrix(t)

    _, _, e_g = geometric_measure_pure(C)

    e_g_curve.append(e_g)

plt.figure(figsize=(8, 5), dpi=300)

plt.plot(np.rad2deg(theta), e_g_curve, linewidth=2,
         label=r'Geometric measure $E_G$')

plt.axvline(x=45, linestyle='--',
            label=r'Maximal entanglement ($\theta=45^\circ$)')

plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel(r'$E_G(\psi)$')
plt.title('Geometric measure of entanglement versus Schmidt angle')
plt.legend()
plt.grid(alpha=0.3)
import os
os.makedirs('figures/ch03', exist_ok=True)
plt.savefig('figures/ch03/geometric_entanglement.png', dpi=300, bbox_inches='tight')
plt.show()