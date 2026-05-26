
		import numpy as np
		import matplotlib.pyplot as plt

		def quantum_relative_entropy(rho, sigma):
			"""Calculates S(rho || sigma) = Tr(rho log2 rho - rho log2 sigma)."""
			# Calculate S(rho)
			eig_rho = np.linalg.eigvalsh(rho)
			eig_rho = eig_rho[eig_rho > 1e-15]
			s_rho = -np.sum(eig_rho * np.log2(eig_rho))

			# Calculate Tr(rho * log2(sigma))
			# Note: Werner states commute, so we can use their eigenvalues
			val_rho = np.linalg.eigvalsh(rho)
			val_sigma = np.linalg.eigvalsh(sigma)
			tr_rho_log_sigma = np.sum(val_rho * np.log2(val_sigma + 1e-15))

			return -s_rho - tr_rho_log_sigma

		# 1. Parameter setup
		p_values = np.array([0.0, 0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
		rho_pure = np.outer(psi_minus, psi_minus)
		identity_4 = np.eye(4) / 4

		# 2. Table Generation
		print(f"{'p':<6} | {'REE (E_R)':<10}")
		print("-" * 20)

		for p in p_values:
			rho_p = p * rho_pure + (1 - p) * identity_4
			if p <= 1/3:
				er = 0.0
			else:
				# Closest separable state for Werner is at p = 1/3
				sigma_star = (1/3) * rho_pure + (2/3) * identity_4
				er = quantum_relative_entropy(rho_p, sigma_star)
			print(f"{p:<6.2f} | {er:<10.4f}")

		# 3. Visualization
		p_fine = np.linspace(0, 1, 100)
		er_plot = []
		for p in p_fine:
			if p <= 1/3:
				er_plot.append(0.0)
			else:
				rho_p = p * rho_pure + (1 - p) * identity_4
				sigma_star = (1/3) * rho_pure + (2/3) * identity_4
				er_plot.append(quantum_relative_entropy(rho_p, sigma_star))

		plt.figure(figsize=(8, 5), dpi=150)
		plt.plot(p_fine, er_plot, 'm-', linewidth=2, label='Relative Entropy of Entanglement')
		plt.axvline(x=1/3, color='k', linestyle='--', alpha=0.5, label='Separability Bound (1/3)')
		plt.xlabel('Purity Parameter (p)')
		plt.ylabel('Entanglement (Bits)')
		plt.title('Relative Entropy of Entanglement in Werner State')
		plt.legend()
		plt.grid(True, alpha=0.2)
		plt.show()