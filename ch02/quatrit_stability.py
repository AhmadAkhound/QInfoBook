		import numpy as np
		import pandas as pd
		import matplotlib.pyplot as plt

		def calculate_qudit_measures(d, p_values):
			"""
			Ab initio calculation of entanglement for d*d Werner states.
			Calculates Negativity and identifies PPT/NPT regions.
			"""
			results = []

			# 1. Construct the d-dimensional Maximally Entangled State (MES)
			phi_d = np.zeros((d**2, 1), dtype=complex)
			for i in range(d):
				basis_vector = np.zeros(d**2)
				basis_vector[i*d + i] = 1
				phi_d += basis_vector.reshape(-1, 1)
			phi_d = phi_d / np.sqrt(d)

			# Projector onto the MES and Maximally mixed state
			rho_phi = np.outer(phi_d, phi_d.conj())
			identity_normalized = np.eye(d**2) / (d**2)

			for p in p_values:
				# 2. Form the Werner State
				rho = p * rho_phi + (1 - p) * identity_normalized

				# 3. Partial Transpose (PT) on the second subsystem
				rho_reshaped = rho.reshape(d, d, d, d)
				rho_pt = rho_reshaped.transpose(0, 3, 2, 1).reshape(d**2, d**2)

				# 4. Negativity Computation via spectrum analysis
				eigenvalues = np.linalg.eigvalsh(rho_pt)
				negative_evs = eigenvalues[eigenvalues < -1e-15]
				negativity = np.sum(np.abs(negative_evs))

				results.append({
					'p': round(p, 4),
					'Negativity': round(negativity, 6),
					'Dimension': d
				})

			return pd.DataFrame(results)

		# --- Execution and Visualization ---
		p_range = np.linspace(0, 1, 11)

		# Qutrit (d=3) and Quatrit (d=4) simulations
		df_d3 = calculate_qudit_measures(3, p_range)
		df_d4 = calculate_qudit_measures(4, p_range)

		# Visualization
		plt.figure(figsize=(8, 5))
		plt.plot(df_d3['p'], df_d3['Negativity'], 'o-', label='Qutrit (d=3)')
		plt.plot(df_d4['p'], df_d4['Negativity'], 's--', label='Quatrit (d=4)')
		plt.axhline(0, color='black', lw=1)
		plt.title("Entanglement Negativity vs Purity (p)")
		plt.xlabel("Purity Parameter (p)")
		plt.ylabel("Negativity (N)")
		plt.legend()
		plt.grid(True)
		plt.show()

		# Display results
		print("Qutrit Results:\n", df_d3.to_string(index=False))
		print("\nQuatrit Results:\n", df_d4.to_string(index=False))