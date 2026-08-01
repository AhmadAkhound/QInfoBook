"""
========================================================================
Title       : Residual Tangle and Three-Way Entanglement
Book        : Quantum Entanglement, Volume 1
Chapter     : 5  --  Uncertainty Relations, Witnesses, and Multipartite Entanglement
Code Ref    : residual_tangle
Description : Computes the residual tangle of three-qubit states from the
              Coffman-Kundu-Wootters monogamy relation, quantifying genuine
              three-way entanglement.
========================================================================
"""
import numpy as np

sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
SY2 = np.kron(sy, sy)

# ---------- three-qubit states ----------
def ghz_state():
    v = np.zeros(8, dtype=complex)
    v[0b000] = 1/np.sqrt(2); v[0b111] = 1/np.sqrt(2)
    return v

def w_state():
    v = np.zeros(8, dtype=complex)
    v[0b100] = 1/np.sqrt(3); v[0b010] = 1/np.sqrt(3); v[0b001] = 1/np.sqrt(3)
    return v

# ---------- reduced density matrices ----------
def reduced_one(state_vec, keep):
    """rho of a single qubit 'keep' (0=A,1=B,2=C)."""
    psi = state_vec.reshape(2, 2, 2)
    axes = [0, 1, 2]; axes.remove(keep)
    rho = np.tensordot(psi, psi.conj(), axes=(axes, axes))
    return rho

def reduced_pair(state_vec, traced):
    """rho of the two qubits, tracing out qubit 'traced'."""
    psi = state_vec.reshape(2, 2, 2)
    rho = np.tensordot(psi, psi.conj(), axes=([traced], [traced]))
    # remaining indices: (i,j) bra (k,l) -> reorder to 4x4
    rho = rho.transpose(0, 1, 2, 3).reshape(4, 4)
    return rho

# ---------- Wootters concurrence ----------
def concurrence(rho):
    R = rho @ (SY2 @ rho.conj() @ SY2)
    ev = np.sort(np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)), 0)))[::-1]
    return max(0.0, ev[0] - ev[1] - ev[2] - ev[3])

# ---------- tangles and CKW check (A as the focus qubit) ----------
def residual_tangle(state_vec):
    rho_A = reduced_one(state_vec, keep=0)
    tau_A_BC = 4 * np.real(np.linalg.det(rho_A))     # A vs (BC)
    rho_AB = reduced_pair(state_vec, traced=2)       # trace C
    rho_AC = reduced_pair(state_vec, traced=1)       # trace B
    tau_AB = concurrence(rho_AB) ** 2
    tau_AC = concurrence(rho_AC) ** 2
    tau_ABC = tau_A_BC - tau_AB - tau_AC
    return tau_A_BC, tau_AB, tau_AC, tau_ABC

print(f"{'state':>5} {'tau_A(BC)':>11} {'tau_AB':>9} {'tau_AC':>9} "
      f"{'tau_ABC':>9} {'CKW ok':>8}")
for name, st in [("GHZ", ghz_state()), ("W", w_state())]:
    tABC_full, tAB, tAC, tABC = residual_tangle(st)
    ckw = (tAB + tAC) <= tABC_full + 1e-9
    print(f"{name:>5} {tABC_full:>11.4f} {tAB:>9.4f} {tAC:>9.4f} "
          f"{tABC:>9.4f} {str(ckw):>8}")