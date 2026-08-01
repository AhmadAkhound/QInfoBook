"""
========================================================================
Title       : Partial Trace and Reduced Density Matrices
Book        : Quantum Entanglement, Volume 1
Chapter     : 1  --  Hilbert Space Foundations and Mathematical Tools
Code Ref    : partial_trace
Description : Builds the partial trace of a bipartite density matrix from
              its definition and compares the reduced states and purities
              of a product state and a maximally entangled Bell state.
========================================================================
"""
import numpy as np

def partial_trace(rho, dims, keep):
	# Partial trace of a bipartite density matrix.
	# rho: full density matrix (dA*dB, dA*dB)
	# dims: [dA, dB] subsystem dimensions
	# keep: 0 to keep subsystem A, 1 to keep subsystem B
	dA, dB = dims
	rho_t = rho.reshape(dA, dB, dA, dB)
	if keep == 0:
		# trace out B: sum over the second and fourth indices
		return np.trace(rho_t, axis1=1, axis2=3)
	else:
		# trace out A: sum over the first and third indices
		return np.trace(rho_t, axis1=0, axis2=2)

	# Computational basis
ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)

	# Product state |0>|+>  (separable)
plus = (ket0 + ket1) / np.sqrt(2)
psi_prod = np.kron(ket0, plus)
rho_prod = psi_prod @ psi_prod.conj().T

	# Bell state |Phi+> = (|00> + |11>)/sqrt(2)  (maximally entangled)
psi_bell = (np.kron(ket0, ket0) + np.kron(ket1, ket1)) / np.sqrt(2)
rho_bell = psi_bell @ psi_bell.conj().T

for name, rho in [("Product |0>|+>", rho_prod), ("Bell |Phi+>", rho_bell)]:
	rhoA = partial_trace(rho, [2, 2], keep=0)
	rhoB = partial_trace(rho, [2, 2], keep=1)
	print(f"=== {name} ===")
	print("rho_A =")
	print(np.round(rhoA, 4))
	print("rho_B =")
	print(np.round(rhoB, 4))
	print("Tr(rho_A) =", np.round(np.trace(rhoA).real, 4))
	print("Tr(rho_A^2) =", np.round(np.trace(rhoA @ rhoA).real, 4))
	print()