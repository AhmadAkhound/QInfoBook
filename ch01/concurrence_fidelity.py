
		import numpy as np
		from scipy.linalg import eigvals

		def calculate_concurrence(rho):
			# Spin-flip operator for two qubits
			sy = np.array([[0, -1j], [1j, 0]])
			sy_sy = np.kron(sy, sy)

			# R matrix construction
			rho_tilde = sy_sy @ rho.conj() @ sy_sy
			R = rho @ rho_tilde

			# Physical eigenvalues extraction
			evals = eigvals(R)
			lams = np.sort(np.sqrt(np.maximum(evals.real, 0)))[::-1]
			return max(0, lams[0] - np.sum(lams[1:]))

		def calculate_fidelity(rho, target_state_vec):
			# Fidelity for mixed state rho and pure target state |psi>: <psi|rho|psi>
			fidelity = target_state_vec.conj().T @ rho @ target_state_vec
			return fidelity.real

		# Simulation Setup
		phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2) # Target Bell state
		rho_pure = np.outer(phi_plus, phi_plus.conj())

		p = 0.8  # Purity
		rho_mixed = p * rho_pure + (1 - p) * np.eye(4) / 4

		# Ab initio results
		con = calculate_concurrence(rho_mixed)
		fid = calculate_fidelity(rho_mixed, phi_plus)

		print(f"Purity (p): {p:.2f}")
		print(f"Calculated Fidelity: {fid:.4f}")
		print(f"Calculated Concurrence: {con:.4f}")