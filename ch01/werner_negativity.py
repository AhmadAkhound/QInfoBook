
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def partial_transpose(rho):
    """Partial transpose with respect to subsystem B."""
    return rho.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)


def trace_distance(rho, sigma):
    """Trace distance: D(rho,sigma)=1/2 ||rho-sigma||_1"""
    delta = rho - sigma
    eigvals = np.linalg.eigvalsh(delta)
    return 0.5 * np.sum(np.abs(eigvals))


def negativity_from_pt(rho_pt):
    """Compute negativity from the eigenvalues of the partial transpose."""
    eigvals = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigvals[eigvals < 0]))


def analyze_werner(p_values):

    results = []

    # Bell singlet state |psi->
    singlet = np.array([0, 1, -1, 0], dtype=float) / np.sqrt(2)
    rho_singlet = np.outer(singlet, singlet)

    # Identity operator and maximally mixed state
    identity = np.eye(4)
    maximally_mixed = identity / 4

    for p in p_values:

        # Construct the Werner state
        rho = p * rho_singlet + (1 - p) * maximally_mixed

        # Partial transpose with respect to subsystem B
        rho_pt = partial_transpose(rho)

        # Eigenvalues of the partial transpose
        pt_eigs = np.linalg.eigvalsh(rho_pt)

        # Smallest eigenvalue
        min_ev = np.min(pt_eigs)

        # Negativity
        negativity = negativity_from_pt(rho_pt)

        # Trace distance from the maximally mixed state
        td = trace_distance(rho, maximally_mixed)

        status = "Entangled (NPT)" if min_ev < -1e-12 else "Separable (PPT)"

        results.append([
            p,
            min_ev,
            negativity,
            td,
            status
        ])

    columns = [
        "p",
        "Min Eigenvalue",
        "Negativity",
        "Trace Distance",
        "Status"
    ]

    return pd.DataFrame(results, columns=columns)


# ---------------------------------------------------
# Generate numerical results
# ---------------------------------------------------

p_range = [0.0, 0.2, 1/3, 0.4, 0.6, 0.8, 1.0]

output_df = analyze_werner(p_range)

print(output_df.to_string(index=False))


# ---------------------------------------------------
# Generate the negativity curve
# ---------------------------------------------------

p = np.linspace(0, 1, 300)

negativity = []

# Bell singlet state
singlet = np.array([0, 1, -1, 0], dtype=float) / np.sqrt(2)
rho_singlet = np.outer(singlet, singlet)

for x in p:

    # Werner state
    rho = x * rho_singlet + (1 - x) * np.eye(4) / 4

    # Partial transpose
    rho_pt = partial_transpose(rho)

    negativity.append(negativity_from_pt(rho_pt))


plt.figure(figsize=(8,5), dpi=300)

plt.plot(
    p,
    negativity,
    linewidth=2,
    label=r'Negativity $\mathcal{N}(\rho_W)$'
)

plt.axvline(
    x=1/3,
    linestyle='--',
    label='Critical Limit ($p=1/3$)'
)

plt.fill_between(
    p,
    0,
    negativity,
    where=(p > 1/3),
    alpha=0.15
)

plt.xlabel("Visibility Parameter (p)")
plt.ylabel("Negativity")
plt.title("Quantum Phase Transition in Werner State")
plt.grid(alpha=0.3)
plt.legend()

plt.savefig(
    "werner_final_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()