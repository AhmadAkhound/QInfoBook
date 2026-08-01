"""
========================================================================
Title       : Entanglement Sudden Death under Amplitude Damping
Book        : Quantum Entanglement, Volume 1
Chapter     : 3  --  Geometric Measures and the Partial Transpose
Code Ref    : decoherence_esd
Description : Builds a two-qubit X-state from its four populations and
              two coherences, evolves it under two independent amplitude
              damping channels using the Kraus operators directly, and
              extracts the negativity from the exact eigenvalues of the
              partial transpose. The closed-form prediction
              N(t) = (1-r) * max(0, w - n4*r) is checked against the
              exact numerical eigenvalues at every time step.
========================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures/ch03", exist_ok=True)


def kraus_amplitude_damping(r):
    E0 = np.array([[1, 0], [0, np.sqrt(1 - r)]])
    E1 = np.array([[0, np.sqrt(r)], [0, 0]])
    return [E0, E1]


def evolve_two_qubit(rho0, r):
    Es = kraus_amplitude_damping(r)
    rho_t = np.zeros_like(rho0, dtype=complex)
    for Ei in Es:
        for Ej in Es:
            K = np.kron(Ei, Ej)
            rho_t += K @ rho0 @ K.conj().T
    return rho_t


def partial_transpose_B(rho):
    rho_reshaped = rho.reshape(2, 2, 2, 2)
    pt = rho_reshaped.transpose(0, 3, 2, 1)
    return pt.reshape(4, 4)


def negativity_from_eigs(rho):
    eigs = np.linalg.eigvalsh(partial_transpose_B(rho))
    return -np.sum(eigs[eigs < 0])


def generate_esd_analysis():
    gamma = 1.0
    t = np.linspace(0, 3, 400)
    r = 1 - np.exp(-gamma * t)

    n1_0, n4_0, w_0 = 0.6, 0.4, 0.3
    assert np.isclose(n1_0 + n4_0, 1.0), "initial state is not normalized"
    assert w_0**2 <= n1_0 * n4_0, "initial state is not positive semi-definite"
    assert n4_0 > w_0, "this choice does not produce sudden death in finite time"

    rho0 = np.zeros((4, 4), dtype=complex)
    rho0[0, 0] = n1_0
    rho0[3, 3] = n4_0
    rho0[0, 3] = w_0
    rho0[3, 0] = w_0

    negativity_exact = np.zeros_like(t)
    negativity_closed = np.zeros_like(t)
    coherence = np.zeros_like(t)

    for k, rk in enumerate(r):
        rho_t = evolve_two_qubit(rho0, rk)
        negativity_exact[k] = negativity_from_eigs(rho_t)
        coherence[k] = rho_t[0, 3].real
        negativity_closed[k] = (1 - rk) * max(0.0, w_0 - n4_0 * rk)

    max_deviation = np.max(np.abs(negativity_exact - negativity_closed))
    print("max |exact - closed_form| over the whole time grid = %.3e" % max_deviation)

    zero_idx = np.where(negativity_exact <= 1e-10)[0]
    t_esd = t[zero_idx[0]] if len(zero_idx) > 0 else None

    end = min(len(t) - 1, zero_idx[0] + 40) if len(zero_idx) else len(t) - 1
    idx = np.linspace(0, end, 8, dtype=int)
    table = pd.DataFrame({
        "gamma_t": t[idx],
        "r": r[idx],
        "coherence": coherence[idx],
        "negativity": negativity_exact[idx],
    })
    table["status"] = table["negativity"].apply(
        lambda x: "Entangled" if x > 1e-10 else "ESD")
    print("\nESD analysis for the X-state under amplitude damping:")
    print(table.to_string(index=False, float_format=lambda x: "%.4f" % x))
    if t_esd is not None:
        print("\nsudden-death time gamma_t = %.4f" % t_esd)
        print("predicted sudden-death r_c = w/n4 = %.4f" % (w_0 / n4_0))

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(t, coherence, "b--", lw=1.5, label="coherence (local)")
    plt.plot(t, negativity_exact, "r-", lw=2.5, label="negativity (entanglement)")
    if t_esd is not None:
        plt.axvline(x=t_esd, color="green", ls=":", label="ESD time")
    plt.fill_between(t, 0, negativity_exact, color="red", alpha=0.1)
    plt.xlabel("dimensionless time (gamma t)")
    plt.ylabel("magnitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("figures/ch03/decoherence_esd.png", dpi=300, bbox_inches="tight")
    plt.show()


generate_esd_analysis()