
		import numpy as np

		def calculate_concurrence_d3(state_vector):
			d = 3
			# Step 1: Ensure normalization
			state_vector = state_vector / np.linalg.norm(state_vector)

			# Step 2: In Schmidt basis, eigenvalues of reduced density matrix
			# are the squared magnitudes of the coefficients.
			rho_red_eigvals = np.abs(state_vector)**2

			# Step 3: Calculate Linear Entropy (1 - Tr(rho_red^2))
			tr_rho_sq = np.sum(rho_red_eigvals**2)
			linear_entropy = 1 - tr_rho_sq

			# Step 4: Generalized Concurrence Formula
			c_d = np.sqrt((d / (d - 1)) * linear_entropy)
			return c_d

		# Physical scenarios for d=3
		states = {
			"Separable": np.array([1, 0, 0]),
			"Entangled in 2D": np.array([1/np.sqrt(2), 1/np.sqrt(2), 0]),
			"Non-uniform Distribution": np.array([0.804, 0.503, 0.318]),
			"Maximally Entangled": np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
		}

		print(f"{'Physical State':<25} | {'Calculated C_d':<15}")
		print("-" * 45)
		for name, vec in states.items():
			c_val = calculate_concurrence_d3(vec)
			print(f"{name:<25} | {c_val:<15.4f}")