"""
========================================================================
Title       : GHZ and W Classes of Tripartite Entanglement
Book        : Quantum Entanglement, Volume 1
Chapter     : 5  --  Uncertainty Relations, Witnesses, and Multipartite Entanglement
Code Ref    : ghzw_classes
Description : Distinguishes the GHZ and W classes of genuine tripartite
              entanglement through their two-qubit reduced states and their
              behaviour under the loss of one qubit.
========================================================================
"""
import numpy as np

# ---------- pauli, spin-flip tools ----------
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
SY2 = np.kron(sy, sy)                       # sigma_y x sigma_y on 2 qubits

# ---------- three-qubit GHZ and W states ----------
def ghz_state():
    v = np.zeros(8, dtype=complex)
    v[0b000] = 1/np.sqrt(2)
    v[0b111] = 1/np.sqrt(2)
    return v

def w_state():
    v = np.zeros(8, dtype=complex)
    v[0b100] = 1/np.sqrt(3)
    v[0b010] = 1/np.sqrt(3)
    v[0b001] = 1/np.sqrt(3)
    return v

# ---------- partial trace of a 3-qubit pure state over qubit C ----------
def reduced_AB(state_vec):
    """rho_AB = Tr_C(|psi><psi|) for a 3-qubit state, qubits ordered A,B,C."""
    psi = state_vec.reshape(2, 2, 2)        # indices A, B, C
    rho_AB = np.zeros((4, 4), dtype=complex)
    for c in range(2):
        block = psi[:, :, c].reshape(4)     # fix C = c -> vector over A,B
        rho_AB += np.outer(block, block.conj())
    return rho_AB

# ---------- Wootters concurrence for a 2-qubit density matrix ----------
def concurrence(rho):
    rho_tilde = SY2 @ rho.conj() @ SY2
    R = rho @ rho_tilde
    ev = np.linalg.eigvals(R)
    ev = np.sort(np.sqrt(np.maximum(np.real(ev), 0)))[::-1]
    return max(0.0, ev[0] - ev[1] - ev[2] - ev[3])

# ---------- one-qubit reduced state and its purity (linear entropy) ----------
def reduced_A(state_vec):
    psi = state_vec.reshape(2, 4)           # A vs (B,C)
    return psi @ psi.conj().T               # rho_A (2x2)

def linear_entropy(rho1):
    return float(np.real(1 - np.trace(rho1 @ rho1)))

# ---------- report for both classes ----------
for name, st in [("GHZ", ghz_state()), ("W", w_state())]:
    rho_AB = reduced_AB(st)
    C_AB = concurrence(rho_AB)
    SL = linear_entropy(reduced_A(st))
    status = "Entangled" if C_AB > 1e-9 else "Separable"
    print(f"{name:>4}: C(rho_AB) = {C_AB:.4f}   "
          f"linear_entropy(rho_A) = {SL:.4f}   reduced AB -> {status}")