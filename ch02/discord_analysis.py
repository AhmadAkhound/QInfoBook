		import numpy as np
		import matplotlib.pyplot as plt
		from scipy.optimize import minimize

		def entropy(rho):
			"""Calculates Von Neumann Entropy."""
			evs = np.linalg.eigvalsh(rho)
			evs = evs[evs > 1e-12]
			return -np.sum(evs * np.log2(evs))

		def werner_state(p):
			"""Generates 2-qubit Werner state."""
			identity = np.eye(4) / 4
			psi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
			rho_bell = np.outer(psi_plus, psi_plus)
			return p * rho_bell + (1 - p) * identity

		def quantum_discord(p):
			rho = werner_state(p)

			# 1. Total Correlation (Mutual Information)
			rho_a = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
			rho_b = np.trace(rho.reshape(2, 2, 2, 2), axis1=0, axis2=2)
			total_corr = entropy(rho_a) + entropy(rho_b) - entropy(rho)

			# 2. Classical Correlation (Optimization over local measurements on B)
			def classical_func(params):
				theta, phi = params
				# Projection operators for qubit B in Bloch sphere representation
				v = np.array([np.cos(theta), np.exp(1j*phi)*np.sin(theta)])
				v_perp = np.array([np.sin(theta), -np.exp(1j*phi)*np.cos(theta)])

				proj = [np.outer(v, v.conj()), np.outer(v_perp, v_perp.conj())]

				cond_entropy = 0
				for P in proj:
					M = np.kron(np.eye(2), P)
					rho_reduced = M @ rho @ M.conj().T
					prob = np.trace(rho_reduced).real
					if prob > 1e-12:
						rho_a_cond = np.trace(rho_reduced.reshape(2, 2, 2, 2), axis1=1, axis2=3) / prob
						cond_entropy += prob * entropy(rho_a_cond)
				return cond_entropy

			res = minimize(classical_func, [0.5, 0.5], bounds=[(0, np.pi), (0, 2*np.pi)])
			max_classical_corr = entropy(rho_a) - res.fun

			discord = total_corr - max_classical_corr
			return total_corr, max_classical_corr, max(0, discord)

		# Execution and Data Generation
		p_values = np.linspace(0, 1, 15)
		results = [quantum_discord(p) for p in p_values]

		# Formatting Table Output
		print(f"{'p (Purity)':<12} | {'Total Corr':<12} | {'Classical':<12} | {'Discord':<12}")
		print("-" * 60)
		for p, (t, c, d) in zip(p_values, results):
			print(f"{p:<12.4f} | {t:<12.4f} | {c:<12.4f} | {d:<12.4f}")

		# Plotting
		p_fine = np.linspace(0, 1, 50)
		results_fine = np.array([quantum_discord(p) for p in p_fine])

		plt.figure(figsize=(8, 5), dpi=150)
		plt.plot(p_fine, results_fine[:, 0], 'k-', label='Total Correlation')
		plt.plot(p_fine, results_fine[:, 1], 'b--', label='Classical Correlation')
		plt.plot(p_fine, results_fine[:, 2], 'r-', lw=2, label='Quantum Discord')
		plt.axvline(x=1/3, color='gray', linestyle=':', label='Entanglement Boundary (p=1/3)')
		plt.xlabel('Purity parameter (p)')
		plt.ylabel('Correlation Measures (Bits)')
		plt.title('Separation of Correlations in Werner State')
		plt.legend()
		plt.grid(alpha=0.3)
		plt.show()