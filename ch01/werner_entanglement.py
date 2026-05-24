
		import numpy as np
		import matplotlib.pyplot as plt
		from scipy.linalg import eigvals

		def get_physical_measures(p):
			# 1. ab initio construction of Werner state
			phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
			rho_phi = np.outer(phi_plus, phi_plus.conj())
			rho_w = p * rho_phi + (1 - p) * np.eye(4) / 4

			# 2. Wootters Spin-flip operation
			sy = np.array([[0, -1j], [1j, 0]])
			sy_sy = np.kron(sy, sy)
			rho_tilde = sy_sy @ rho_w.conj() @ sy_sy

			# 3. Eigenvalues of R = rho * rho_tilde
			R = rho_w @ rho_tilde
			evals = eigvals(R)
			roots = np.sort(np.sqrt(np.maximum(evals.real, 0)))[::-1]

			# 4. Calculation of Measures
			c = max(0, roots[0] - np.sum(roots[1:]))
			return c, c**2

		# Outputting Table Data
		p_vals = [0.20, 0.33, 0.40, 0.60, 0.80, 1.00]
		print(f"{'p':<8} | {'C':<10} | {'tau':<10} | {'Status'}")
		print("-" * 45)
		for p in p_vals:
			c, tau = get_physical_measures(p)
			status = "Separable" if c < 1e-6 else "Entangled"
			print(f"{p:<8.2f} | {c:<10.4f} | {tau:<10.4f} | {status}")

		# Plotting High-Quality Figure
		p_fine = np.linspace(0, 1, 500)
		results = [get_physical_measures(p) for p in p_fine]
		c_fine, tau_fine = zip(*results)

		plt.figure(figsize=(8, 6), dpi=100)
		plt.plot(p_fine, c_fine, 'r-', label='Concurrence (C) - Linear', linewidth=2)
		plt.plot(p_fine, tau_fine, 'b--', label='Tangle (tau) - Parabolic', linewidth=2)
		plt.axvline(x=1/3, color='k', linestyle=':', label='Threshold (p=1/3)')
		plt.xlabel('Purity Parameter (p)')
		plt.ylabel('Measure Value')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.show()