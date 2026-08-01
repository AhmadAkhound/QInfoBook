"""
========================================================================
Title       : Stabilizer Entanglement Witness for the GHZ State
Book        : Quantum Entanglement, Volume 1
Chapter     : 5  --  Uncertainty Relations, Witnesses, and Multipartite Entanglement
Code Ref    : stabilizer_witness
Description : Constructs a stabilizer-based entanglement witness for the GHZ
              state and evaluates its expectation value to detect genuine
              multipartite entanglement.
========================================================================
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)

# ---------- three-qubit GHZ state and its projector ----------
def ghz_state():
    v = np.zeros(8, dtype=complex)
    v[0b000] = 1/np.sqrt(2); v[0b111] = 1/np.sqrt(2)
    return v

ghz = ghz_state()
P_ghz = np.outer(ghz, ghz.conj())

# ---------- noisy GHZ: white noise mixture ----------
def noisy_ghz(pn):
    """(1-pn) |GHZ><GHZ| + pn * I/8 ; pn = white-noise fraction."""
    return (1 - pn) * P_ghz + pn * np.eye(8, dtype=complex) / 8

# ---------- stabilizer witness W = (1/2) I - |GHZ><GHZ| ----------
W_ghz = 0.5 * np.eye(8, dtype=complex) - P_ghz

def expval(W, rho):
    return float(np.real(np.trace(W @ rho)))

# ---------- check the two stabilizers have eigenvalue +1 on GHZ ----------
S1 = kron3(sx, sx, sx)
S2 = kron3(sz, sz, I2)
print("Stabilizer check on |GHZ>:")
print(f"  <GHZ|S1|GHZ> = {expval(S1, P_ghz):.4f}")
print(f"  <GHZ|S2|GHZ> = {expval(S2, P_ghz):.4f}")

# ---------- witness expectation vs white-noise fraction ----------
print("\nStabilizer witness on noisy GHZ:")
print(f"{'p_noise':>8} {'Tr(W rho)':>12} {'GME detected':>14}")
for pn in [0.0, 0.1, 0.2, 0.3, 0.4, 0.4286, 0.5, 0.6]:
    val = expval(W_ghz, noisy_ghz(pn))
    print(f"{pn:>8.4f} {val:>12.4f} {str(val < -1e-9):>14}")