
		import numpy as np
		import matplotlib.pyplot as plt
		from scipy.linalg import sqrtm

		def calculate_fidelity(rho, sigma):
			"""Calculates Fidelity between two density matrices rho and sigma."""
			try:
				# F(rho, sigma) = [Tr(sqrt(sqrt(rho) * sigma * sqrt(rho)))]^2
				sqrt_rho = sqrtm(rho)
				matrix_product = sqrt_rho @ sigma @ sqrt_rho
				fidelity_root = np.trace(sqrtm(matrix_product)).real
				return fidelity_root**2
			except (ValueError, np.linalg.LinAlgError):
				# Fallback for numerical stability in edge cases
				return np.abs(np.trace(rho @ sigma))

		# 1. Parameter setup
		p_values = np.array([0.0, 0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		p_fine = np.linspace(0, 1, 100)

		# Target state: Pure Bell State |Psi->
		psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
		rho_target = np.outer(psi_minus, psi_minus)
		identity_4 = np.eye(4) / 4

		# 2. Numerical Table Generation
		print(f"{'p':<6} | {'Fidelity (F)':<12}")
		print("-" * 22)

		for p in p_values:
			rho_werner = p * rho_target + (1 - p) * identity_4
			fid = calculate_fidelity(rho_werner, rho_target)
			status = "<-- Threshold" if abs(fid - 0.5) < 0.01 else ""
			print(f"{p:<6.2f} | {fid:<12.4f} {status}")

		# 3. Visualization
		f_plot = []
		for p in p_fine:
			rho_w = p * rho_target + (1 - p) * identity_4
			f_plot.append(calculate_fidelity(rho_w, rho_target))

		plt.figure(figsize=(8, 5), dpi=150)
		# Using raw string (r'') for LaTeX compatibility in labels
		plt.plot(p_fine, f_plot, 'r-', linewidth=2, label=r'Fidelity with $|\Psi^-\rangle$')
		plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.6, label='Classical Limit (0.5)')
		plt.axvline(x=1/3, color='blue', linestyle=':', label='Separability Bound (p=1/3)')

		# Shading the entanglement region
		plt.fill_between(p_fine, 0.5, f_plot, where=(np.array(f_plot) > 0.5),
					  	color='green', alpha=0.1, label='Entangled Region')

		plt.xlabel('Purity Parameter (p)')
		plt.ylabel('Fidelity')
		plt.title('Fidelity as a Quality Metric in Werner States')
		plt.legend(loc='lower right')
		plt.grid(True, alpha=0.2)
		plt.show()