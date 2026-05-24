
		import numpy as np
		from scipy.linalg import sqrtm

		def get_fidelity(rho, sigma):
			# Ab initio calculation with stability check
			# Purity check to avoid singular matrix warnings
			purity_rho = np.real(np.trace(rho @ rho))

			if purity_rho > 0.99:
				# Simplified formula for pure states: F = tr(rho * sigma)
				return np.real(np.trace(rho @ sigma))
			else:
				# General Jozsa-Uhlmann formula for mixed states
				sq_rho = sqrtm(rho)
				temp = sq_rho @ sigma @ sq_rho
				return np.real(np.trace(sqrtm(temp)))**2

		# 1. Target Bell State |phi+>
		phi_p = np.array([1, 0, 0, 1]) / np.sqrt(2)
		rho_bell = np.outer(phi_p, phi_p.conj())

		# 2. Mixed State (Identity noise)
		rho_mixed = np.eye(4) / 4

		# 3. Orthogonal State (Singlet state |psi->)
		psi_o = np.array([0, 1, -1, 0]) / np.sqrt(2)
		rho_ortho = np.outer(psi_o, psi_o.conj())

		# Numerical Results
		f_mixed = get_fidelity(rho_bell, rho_mixed)
		f_ortho = get_fidelity(rho_bell, rho_ortho)

		print(f"Fidelity (Bell vs Mixed): {f_mixed:.4f}")
		print(f"Fidelity (Bell vs Ortho): {f_ortho:.4f}")
		print(f"Is Entangled (F > 0.5): {f_mixed > 0.5}")