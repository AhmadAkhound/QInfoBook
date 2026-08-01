"""
========================================================================
Title       : Quantum Discord and Classical Correlation in Werner States
Book        : Quantum Entanglement, Volume 1
Chapter     : 7  --  Continuous Variables and Correlations Beyond Entanglement
Code Ref    : discord_analysis
Description : Computes the total mutual information, classical correlation,
              quantum discord, and one-way quantum deficit of the Werner state
              as a function of the mixing parameter.
========================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def entropy(rho):
    """Von Neumann entropy in bits."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return -np.sum(eigenvalues * np.log2(eigenvalues))


def werner_state(p):
    """Two-qubit Werner state: Bell state mixed with the maximally mixed state."""
    identity = np.eye(4, dtype=complex) / 4
    psi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho_bell = np.outer(psi_plus, psi_plus.conj())
    return p * rho_bell + (1 - p) * identity


def computational_dephasing_entropy_increase(p):
    """
    Entropy increase produced by local computational-basis dephasing.

    For the Werner family considered here, symmetry ensures that this value
    equals the optimized one-way quantum deficit. This function itself does
    not perform the minimization over all local projective measurements.
    """
    rho = werner_state(p)

    projectors = [
        np.array([[1, 0], [0, 0]], dtype=complex),
        np.array([[0, 0], [0, 1]], dtype=complex),
    ]

    rho_dephased = np.zeros_like(rho, dtype=complex)

    for projector in projectors:
        measurement_operator = np.kron(np.eye(2), projector)
        rho_dephased += (
            measurement_operator
            @ rho
            @ measurement_operator.conj().T
        )

    return max(0.0, entropy(rho_dephased) - entropy(rho))


def quantum_discord(p):
    """Return total, classical, discord, and deficit correlations."""
    rho = werner_state(p)

    # Reduced density matrices
    rho_tensor = rho.reshape(2, 2, 2, 2)
    rho_a = np.trace(rho_tensor, axis1=1, axis2=3)
    rho_b = np.trace(rho_tensor, axis1=0, axis2=2)

    # Total correlation: quantum mutual information
    total_corr = entropy(rho_a) + entropy(rho_b) - entropy(rho)

    # Conditional entropy after a projective measurement on subsystem B
    def conditional_entropy(params):
        theta, phi = params

        vector_0 = np.array(
            [
                np.cos(theta),
                np.exp(1j * phi) * np.sin(theta),
            ],
            dtype=complex,
        )

        vector_1 = np.array(
            [
                np.sin(theta),
                -np.exp(1j * phi) * np.cos(theta),
            ],
            dtype=complex,
        )

        projectors = [
            np.outer(vector_0, vector_0.conj()),
            np.outer(vector_1, vector_1.conj()),
        ]

        average_conditional_entropy = 0.0

        for projector in projectors:
            measurement_operator = np.kron(np.eye(2), projector)

            post_measurement_term = (
                measurement_operator
                @ rho
                @ measurement_operator.conj().T
            )

            probability = np.trace(post_measurement_term).real

            if probability > 1e-12:
                conditional_state_a = np.trace(
                    post_measurement_term.reshape(2, 2, 2, 2),
                    axis1=1,
                    axis2=3,
                ) / probability

                average_conditional_entropy += (
                    probability * entropy(conditional_state_a)
                )

        return average_conditional_entropy

    # Multiple starting points reduce dependence on a single local search
    starts = [
        (0.0, 0.0),
        (np.pi / 4, 0.0),
        (np.pi / 4, np.pi / 2),
        (np.pi / 2, 0.0),
    ]

    best_conditional_entropy = np.inf

    for start in starts:
        result = minimize(
            conditional_entropy,
            start,
            method="L-BFGS-B",
            bounds=[
                (0.0, np.pi),
                (0.0, 2.0 * np.pi),
            ],
        )

        if not result.success:
            raise RuntimeError(
                "Projective-measurement optimization failed: "
                + result.message
            )

        best_conditional_entropy = min(
            best_conditional_entropy,
            result.fun,
        )

    classical_corr = entropy(rho_a) - best_conditional_entropy
    classical_corr = max(0.0, classical_corr)

    discord = max(0.0, total_corr - classical_corr)

    deficit = computational_dephasing_entropy_increase(p)

    return total_corr, classical_corr, discord, deficit


# Values used in the numerical table
p_values = np.linspace(0, 1, 15)
results = [quantum_discord(p) for p in p_values]

print(
    "%-10s | %-10s | %-10s | %-10s | %-10s"
    % ("p", "Total", "Classical", "Discord", "Deficit")
)
print("-" * 62)

for p, (total, classical, discord, deficit) in zip(p_values, results):
    print(
        "%-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f"
        % (p, total, classical, discord, deficit)
    )


# Values used in the figure
p_fine = np.linspace(0, 1, 50)
results_fine = np.array(
    [quantum_discord(p) for p in p_fine]
)

plt.figure(figsize=(8, 5), dpi=150)

plt.plot(
    p_fine,
    results_fine[:, 0],
    "k-",
    label="Total correlation",
)

plt.plot(
    p_fine,
    results_fine[:, 1],
    "b--",
    label="Classical correlation",
)

plt.plot(
    p_fine,
    results_fine[:, 2],
    "r-",
    label="Quantum discord",
)

plt.plot(
    p_fine,
    results_fine[:, 3],
    "g-.",
    linewidth=2,
    label="One-way quantum deficit",
)

plt.axvline(
    x=1 / 3,
    color="gray",
    linestyle=":",
    label="Entanglement boundary",
)

plt.xlabel("Werner parameter p")
plt.ylabel("Correlation measures (bits)")
plt.legend()
plt.grid(alpha=0.3)

os.makedirs("figures/ch07", exist_ok=True)

plt.savefig(
    "figures/ch07/discord_analysis.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()