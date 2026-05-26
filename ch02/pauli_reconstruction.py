		import numpy as np

		def pauli_matrices():
			"""Returns the standard 2x2 Pauli matrices and Identity."""
			I = np.eye(2)
			X = np.array([[0, 1], [1, 0]])
			Y = np.array([[0, -1j], [1j, 0]])
			Z = np.array([[1, 0], [0, -1]])
			return I, X, Y, Z

		def get_werner_state(p):
			"""Generates the Werner state rho = p|psi-><psi-| + (1-p)/4 * I"""
			# |psi-> = 1/sqrt(2) * (|01> - |10>)
			psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
			rho_pure = np.outer(psi_minus, psi_minus)
			return p * rho_pure + (1 - p) / 4 * np.eye(4)

		def simulate_lab_measurement(rho):
			"""
			Simulates the physical measurement of Pauli correlations.
			W = 1/4 * (I*I + sigma_x*sigma_x + sigma_y*sigma_y + sigma_z*sigma_z)
			"""
			I, X, Y, Z = pauli_matrices()

			# Constructing two-qubit operators for correlations
			XX = np.kron(X, X)
			YY = np.kron(Y, Y)
			ZZ = np.kron(Z, Z)

			# Expectation values <sigma_i \otimes \sigma_i>
			exp_XX = np.trace(XX @ rho).real
			exp_YY = np.trace(YY @ rho).real
			exp_ZZ = np.trace(ZZ @ rho).real

			# Reconstructing the witness value from local measurements
			witness_val = 0.25 * (1 + exp_XX + exp_YY + exp_ZZ)
			return witness_val, exp_XX, exp_YY, exp_ZZ

		# --- Main Execution ---
		print(f"{'p':<5} | {'<X@X>':<8} | {'<Z@Z>':<8} | {'Witness':<10} | {'Theoretical'}")
		print("-" * 55)

		p_points = [0.0, 0.33, 0.5, 1.0]
		for p in p_points:
			rho = get_werner_state(p)
			w_calc, ex, ey, ez = simulate_lab_measurement(rho)
			w_theory = (1 - 3*p) / 4

			print(f"{p:<5.2f} | {ex:<8.3f} | {ez:<8.3f} | {w_calc:<10.4f} | {w_theory:<11.4f}")

		# Example analysis for p=0.5
		p_test = 0.5
		rho_test = get_werner_state(p_test)
		val, ex, ey, ez = simulate_lab_measurement(rho_test)

		print(f"\nDetailed analysis for p = {p_test}:")
		print(f"Experimental Correlations: X@X={ex:.2f}, Y@Y={ey:.2f}, Z@Z={ez:.2f}")
		print(f"Final Witness Value: {val:.4f}")