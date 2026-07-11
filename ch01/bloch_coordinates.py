"""
========================================================================
Title       : Bloch Vector Coordinates and State Purity
Book        : Quantum Entanglement, Volume 1
Chapter     : 1  --  Hilbert Space Foundations and Mathematical Tools
Code Ref    : bloch_coordinates
Description : Computes the Bloch vector of a single qubit from the Pauli
              expectation values and relates its length to the state
              purity for pure, maximally mixed, and partially mixed states.
========================================================================
"""
import numpy as np

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def bloch_vector(rho):
	# Bloch coordinates from expectation values of Pauli operators
	rx = np.trace(rho @ X).real
	ry = np.trace(rho @ Y).real
	rz = np.trace(rho @ Z).real
	return np.array([rx, ry, rz])

# Sample states
ket0 = np.array([[1], [0]], dtype=complex)
plus = (np.array([[1], [0]], dtype=complex) +
np.array([[0], [1]], dtype=complex)) / np.sqrt(2)

states = {
	"|0>"                : ket0 @ ket0.conj().T,
	"|+>"                : plus @ plus.conj().T,
	"Maximally mixed I/2": I / 2,
	"Partially mixed"    : 0.75 * (ket0 @ ket0.conj().T) + 0.25 * (I / 2),
}

for name, rho in states.items():
	r = bloch_vector(rho)
	length = np.linalg.norm(r)
	purity = np.trace(rho @ rho).real
	print(f"=== {name} ===")
	print("Bloch vector (rx, ry, rz) =", np.round(r, 4))
	print("Length |r| =", np.round(length, 4))
	print("Purity Tr(rho^2) =", np.round(purity, 4))
	print()