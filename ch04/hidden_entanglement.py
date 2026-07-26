"""
========================================================================
Title       : Probabilistic Entanglement Concentration by Local Filtering
Book        : Quantum Entanglement, Volume 1
Chapter     : 4 -- Operational Measures and the Cost of Entanglement
Code Ref    : hidden_entanglement

Description : Computes the concurrence of an asymmetric mixed two-qubit
              state before local filtering and in the successful and failed
              branches of a complete local filtering instrument. It also
              evaluates the success probability and the probability-weighted
              average concurrence.

Important   : An increase of concurrence in the successful branch is a
              conditional, probabilistic increase. The average concurrence
              over all outcomes cannot exceed the initial concurrence.
========================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Basic matrices
# ---------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)

sigma_y = np.array(
    [
        [0, -1j],
        [1j, 0],
    ],
    dtype=complex,
)

YY = np.kron(sigma_y, sigma_y)


# ---------------------------------------------------------------------
# Concurrence
# ---------------------------------------------------------------------

def concurrence(rho):
    """
    Wootters concurrence for a two-qubit density matrix.
    """
    rho_tilde = YY @ rho.conj() @ YY
    product = rho @ rho_tilde

    eigenvalues = np.linalg.eigvals(product)
    eigenvalues = np.clip(eigenvalues.real, 0.0, None)

    roots = np.sort(np.sqrt(eigenvalues))[::-1]

    value = roots[0] - roots[1] - roots[2] - roots[3]
    return max(0.0, float(value))


# ---------------------------------------------------------------------
# State family
# ---------------------------------------------------------------------

def asymmetric_pure_state(alpha, beta):
    """
    |psi> = alpha |00> + beta |11>.
    """
    psi = np.array(
        [
            alpha,
            0.0,
            0.0,
            beta,
        ],
        dtype=complex,
    )

    norm = np.linalg.norm(psi)

    if norm <= 1e-15:
        raise ValueError("The pure-state vector must be nonzero.")

    return psi / norm


def mixed_state(p, alpha, beta):
    """
    rho(p) = p |psi><psi| + (1-p) I_4/4.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must satisfy 0 <= p <= 1.")

    psi = asymmetric_pure_state(alpha, beta)
    projector = np.outer(psi, psi.conj())

    return p * projector + (1.0 - p) * np.eye(4, dtype=complex) / 4.0


# ---------------------------------------------------------------------
# Local filtering instrument
# ---------------------------------------------------------------------

def conditional_output(rho, kraus_operator):
    """
    Return the branch probability and its normalized conditional state.
    """
    unnormalized = (
        kraus_operator
        @ rho
        @ kraus_operator.conj().T
    )

    probability = float(np.trace(unnormalized).real)

    if probability <= 1e-15:
        return 0.0, None

    conditional_state = unnormalized / probability

    return probability, conditional_state


def local_filter_branches(rho, f1, f2):
    """
    Construct a complete two-outcome local filtering instrument.

    Successful operator:
        A = diag(f1, f2)

    Failed operator:
        B = sqrt(I - A^dagger A)

    The operators act on the first qubit:
        K_success = A tensor I
        K_failure = B tensor I
    """
    if not 0.0 <= f1 <= 1.0:
        raise ValueError("f1 must satisfy 0 <= f1 <= 1.")

    if not 0.0 <= f2 <= 1.0:
        raise ValueError("f2 must satisfy 0 <= f2 <= 1.")

    A = np.diag([f1, f2]).astype(complex)

    B = np.diag(
        [
            np.sqrt(max(0.0, 1.0 - f1**2)),
            np.sqrt(max(0.0, 1.0 - f2**2)),
        ]
    ).astype(complex)

    completeness_error = np.linalg.norm(
        A.conj().T @ A
        + B.conj().T @ B
        - I2
    )

    if completeness_error > 1e-12:
        raise RuntimeError("The filter Kraus operators are not complete.")

    K_success = np.kron(A, I2)
    K_failure = np.kron(B, I2)

    p_success, rho_success = conditional_output(
        rho,
        K_success,
    )

    p_failure, rho_failure = conditional_output(
        rho,
        K_failure,
    )

    probability_error = abs(
        p_success + p_failure - 1.0
    )

    if probability_error > 1e-10:
        raise RuntimeError(
            "The branch probabilities do not sum to one."
        )

    return (
        p_success,
        rho_success,
        p_failure,
        rho_failure,
    )


# ---------------------------------------------------------------------
# Parameters used in Chapter 4
# ---------------------------------------------------------------------

alpha = 0.9
beta = np.sqrt(1.0 - alpha**2)

# The larger |00> coefficient is attenuated more strongly.
f1 = beta / alpha
f2 = 1.0

p_values = np.linspace(0.4, 0.9, 7)


# ---------------------------------------------------------------------
# Numerical table
# ---------------------------------------------------------------------

header = (
    f"{'p':<8} | "
    f"{'C_initial':<10} | "
    f"{'p_success':<10} | "
    f"{'C_success':<10} | "
    f"{'p_failure':<10} | "
    f"{'C_failure':<10} | "
    f"{'C_average':<10}"
)

print(header)
print("-" * len(header))

rows = []

for p in p_values:
    rho = mixed_state(
        p,
        alpha,
        beta,
    )

    c_initial = concurrence(rho)

    (
        p_success,
        rho_success,
        p_failure,
        rho_failure,
    ) = local_filter_branches(
        rho,
        f1,
        f2,
    )

    c_success = (
        concurrence(rho_success)
        if rho_success is not None
        else 0.0
    )

    c_failure = (
        concurrence(rho_failure)
        if rho_failure is not None
        else 0.0
    )

    c_average = (
        p_success * c_success
        + p_failure * c_failure
    )

    rows.append(
        (
            p,
            c_initial,
            p_success,
            c_success,
            p_failure,
            c_failure,
            c_average,
        )
    )

    print(
        f"{p:<8.4f} | "
        f"{c_initial:<10.4f} | "
        f"{p_success:<10.4f} | "
        f"{c_success:<10.4f} | "
        f"{p_failure:<10.4f} | "
        f"{c_failure:<10.4f} | "
        f"{c_average:<10.4f}"
    )

    # Entanglement monotonicity on average.
    if c_average > c_initial + 1e-9:
        raise RuntimeError(
            "Average concurrence increased under the local instrument."
        )


# ---------------------------------------------------------------------
# Fine grid for plotting
# ---------------------------------------------------------------------

p_fine = np.linspace(0.3, 0.95, 200)

c_initial_fine = []
c_success_fine = []
c_average_fine = []
p_success_fine = []

for p in p_fine:
    rho = mixed_state(
        p,
        alpha,
        beta,
    )

    c_initial = concurrence(rho)

    (
        p_success,
        rho_success,
        p_failure,
        rho_failure,
    ) = local_filter_branches(
        rho,
        f1,
        f2,
    )

    c_success = (
        concurrence(rho_success)
        if rho_success is not None
        else 0.0
    )

    c_failure = (
        concurrence(rho_failure)
        if rho_failure is not None
        else 0.0
    )

    c_average = (
        p_success * c_success
        + p_failure * c_failure
    )

    c_initial_fine.append(c_initial)
    c_success_fine.append(c_success)
    c_average_fine.append(c_average)
    p_success_fine.append(p_success)


c_initial_fine = np.array(c_initial_fine)
c_success_fine = np.array(c_success_fine)
c_average_fine = np.array(c_average_fine)
p_success_fine = np.array(p_success_fine)


# ---------------------------------------------------------------------
# Locate the physical crossover of initial and successful-branch
# concurrence
# ---------------------------------------------------------------------

difference = c_success_fine - c_initial_fine

# Ignore the separable region and numerical zeros, where both
# concurrences vanish and a spurious sign change may be detected.
tolerance = 1e-8

valid_intervals = (
    (c_initial_fine[:-1] > tolerance)
    & (c_initial_fine[1:] > tolerance)
    & (c_success_fine[:-1] > tolerance)
    & (c_success_fine[1:] > tolerance)
)

crossing_indices = np.where(
    valid_intervals
    & (
        difference[:-1] * difference[1:] <= 0.0
    )
)[0]

if len(crossing_indices) > 0:
    index = crossing_indices[0]

    p_left = p_fine[index]
    p_right = p_fine[index + 1]

    d_left = difference[index]
    d_right = difference[index + 1]

    # Linear interpolation between the two neighbouring grid points.
    if abs(d_right - d_left) > 1e-15:
        p_crossing = (
            p_left
            - d_left * (p_right - p_left)
            / (d_right - d_left)
        )
    else:
        p_crossing = 0.5 * (p_left + p_right)

    print(
        "\nApproximate physical crossing between initial and "
        "successful-branch concurrence: "
        f"p = {p_crossing:.4f}"
    )
else:
    p_crossing = None

    print(
        "\nNo physical crossing was detected in the region where "
        "both concurrences are positive."
    )
# ---------------------------------------------------------------------
# Internal consistency report
# ---------------------------------------------------------------------

max_average_excess = np.max(
    c_average_fine - c_initial_fine
)

print(
    "Maximum value of "
    "C_average - C_initial over the grid: "
    f"{max_average_excess:.3e}"
)

print(
    "The successful branch can have larger concurrence, "
    "but the probability-weighted average does not."
)


# ---------------------------------------------------------------------
# Figure used in Chapter 4
# ---------------------------------------------------------------------

os.makedirs(
    "figures/ch04",
    exist_ok=True,
)

plt.figure(
    figsize=(8, 5),
    dpi=150,
)

plt.plot(
    p_fine,
    c_initial_fine,
    linewidth=2,
    label="Initial concurrence",
)

plt.plot(
    p_fine,
    c_success_fine,
    linestyle="--",
    linewidth=2,
    label="Successful-branch concurrence",
)

plt.plot(
    p_fine,
    c_average_fine,
    linestyle=":",
    linewidth=2,
    label="Probability-weighted average concurrence",
)

plt.xlabel("Purity parameter $p$")
plt.ylabel("Concurrence")

plt.title(
    "Probabilistic entanglement concentration "
    "by local filtering"
)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/ch04/hidden_entanglement.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()