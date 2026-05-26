		import numpy as np
		import matplotlib.pyplot as plt
		import pandas as pd

		def concurrence(rho):
			"""Exact Wootters Concurrence calculation."""
			sigma_y = np.array([[0, -1j], [1j, 0]])
			sy_sy = np.kron(sigma_y, sigma_y)
			rho_tilde = sy_sy @ rho.conj() @ sy_sy
			# Calculation of R = rho * rho_tilde
			R = rho @ rho_tilde
			evs = np.linalg.eigvals(R)
			# Handling numerical precision for square root
			evs = np.sort(np.real(np.sqrt(np.maximum(evs, 0))))[::-1]
			return max(0, evs[0] - evs[1] - evs[2] - evs[3])

		def hidden_entanglement_calc(p, alpha=0.9):
			"""
			p: purity (weight of the pure state)
			alpha: asymmetry of the base state (alpha|00> + beta|11>)
			"""
			# 1. Base non-maximal state: alpha|00> + beta|11>
			beta = np.sqrt(1 - alpha**2)
			psi = np.array([alpha, 0, 0, beta])
			rho_pure = np.outer(psi, psi)

			# 2. Noisy state formation
			rho = p * rho_pure + (1 - p) * (np.eye(4) / 4)
			c_initial = concurrence(rho)

			# 3. Optimal Local Filtering (Procrustean)
			# Balancing coefficients by filtering the first qubit
			f1 = 1.0
			f2 = alpha / beta if alpha < beta else beta / alpha
			if alpha > beta:
				filter_mat = np.array([[f2, 0], [0, f1]])
			else:
				filter_mat = np.array([[f1, 0], [0, f2]])

			F = np.kron(filter_mat, np.eye(2))
			rho_filtered = F @ rho @ F.conj().T
			rho_filtered /= np.trace(rho_filtered) # Re-normalization

			c_filtered = concurrence(rho_filtered)
			return c_initial, c_filtered

		# --- Data Generation ---
		p_values = np.linspace(0.4, 0.9, 15)
		results = [hidden_entanglement_calc(p) for p in p_values]
		c_init, c_filt = zip(*results)

		# --- Numerical Output for LaTeX Table ---
		df = pd.DataFrame({
			'Purity (p)': p_values,
			'Initial Conc': c_init,
			'Filtered Conc': c_filt
		})
		print(df.to_string(index=False))

		# --- Visualization ---
		plt.figure(figsize=(7, 5), dpi=150)
		plt.plot(p_values, c_init, 'b--', label='Initial (Asymmetric)')
		plt.plot(p_values, c_filt, 'r-', lw=2, label='After Optimal Filtering')
		plt.fill_between(p_values, c_init, c_filt, color='red', alpha=0.1)
		plt.xlabel('Purity (p)')
		plt.ylabel('Concurrence')
		plt.title('Activation of Hidden Entanglement')
		plt.legend()
		plt.grid(alpha=0.3)
		plt.show()