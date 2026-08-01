"""
========================================================================
Title       : Construction of an Entanglement Witness
Book        : Quantum Entanglement, Volume 1
Chapter     : 5  --  Uncertainty Relations, Witnesses, and Multipartite Entanglement
Code Ref    : witness_construction
Description : Constructs an entanglement witness from the projector onto a Bell
              state and evaluates its expectation value to distinguish
              entangled from separable states.
========================================================================
"""
import numpy as np

# ---------- pauli operators, bell states ----------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

phi_plus  = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)   # |Phi+>
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)  # |Psi->

P_phi = np.outer(phi_plus,  phi_plus.conj())
P_psi = np.outer(psi_minus, psi_minus.conj())

def werner(p):
    """Werner state (eq. ch2 werner def)."""
    return p * P_phi + (1 - p) * np.eye(4, dtype=complex) / 4

# ---------- partial transpose on subsystem B ----------
def partial_transpose_B(rho):
    r = rho.reshape(2, 2, 2, 2)
    r = r.transpose(0, 3, 2, 1)          # transpose B indices
    return r.reshape(4, 4)

# ---------- witness from the negative PPT eigenvector ----------
def ppt_witness(rho):
    """W = (|eta><eta|)^{T_B}, eta = eigenvector of rho^{T_B} with min eigenvalue."""
    rho_TB = partial_transpose_B(rho)
    vals, vecs = np.linalg.eigh(rho_TB)
    eta = vecs[:, 0]                      # smallest eigenvalue
    P_eta = np.outer(eta, eta.conj())
    return partial_transpose_B(P_eta), vals[0]

# ---------- fidelity witness for a pure target ----------
def fidelity_witness(target, alpha):
    """W = alpha I - |target><target|."""
    P = np.outer(target, target.conj())
    return alpha * np.eye(4, dtype=complex) - P

def expval(W, rho):
    return float(np.real(np.trace(W @ rho)))

# ---------- 1) PPT witness on Werner: threshold scan ----------
print("PPT witness on Werner state:  Tr(W rho) = (1-3p)/4")
print(f"{'p':>5} {'Tr(W rho)':>12} {'detected':>10}")
W_ppt, _ = ppt_witness(werner(0.9))      # build from a clearly entangled member
for p in [0.20, 1/3, 0.40, 0.60, 0.80, 1.0]:
    val = expval(W_ppt, werner(p))
    print(f"{p:>5.3f} {val:>12.4f} {str(val < -1e-9):>10}")

# ---------- 2) fidelity witness on noisy Bell family ----------
# rho(q) = q |Phi+><Phi+| + (1-q) |00><00|
ket00 = np.array([1, 0, 0, 0], dtype=complex)
P00 = np.outer(ket00, ket00.conj())
W_fid = fidelity_witness(phi_plus, 0.5)
print("\nFidelity witness on rho(q) = q|Phi+><Phi+| + (1-q)|00><00|:")
print(f"{'q':>5} {'F(q)':>8} {'Tr(W rho)':>12} {'detected':>10}")
for q in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    rho = q * P_phi + (1 - q) * P00
    F = expval(P_phi, rho)
    val = expval(W_fid, rho)
    print(f"{q:>5.2f} {F:>8.4f} {val:>12.4f} {str(val < -1e-9):>10}")

# ---------- 3) blind spot: Phi+ witness on orthogonal Psi- ----------
print("\nBlind spot test: fidelity witness of |Phi+> applied to |Psi->")
val_blind = expval(W_fid, P_psi)
print(f"Tr(W |Psi-><Psi-|) = {val_blind:.4f}   (positive => not detected)")