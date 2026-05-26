
		import numpy as np
		import matplotlib.pyplot as plt

		def get_von_neumann_entropy(rho):
			"""Calculates Von Neumann entropy for a given density matrix."""
			eigvals = np.linalg.eigvalsh(rho)
			eigvals = np.real(eigvals[eigvals > 1e-15])
			return -np.sum(eigvals * np.log2(eigvals))

		def get_shannon_entropy(probabilities):
			"""Calculates Shannon entropy for a probability distribution."""
			probabilities = probabilities[probabilities > 1e-15]
			return -np.sum(probabilities * np.log2(probabilities))

		# 1. Configuration
		p_values = np.array([0.0, 0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		table_results = []
		H_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
		psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
		rho_pure = np.outer(psi_minus, psi_minus)

		for p in p_values:
			# 2. Construction of Werner State
			rho_ab = p * rho_pure + (1 - p) * np.eye(4) / 4

			# 3. Local Measurement on Qubit A
			rho_a = np.trace(rho_ab.reshape(2, 2, 2, 2), axis1=1, axis2=3)
			prob_z = np.diag(rho_a).real
			rho_a_x = H_gate @ rho_a @ H_gate.T
			prob_x = np.diag(rho_a_x).real
			sum_h = get_shannon_entropy(prob_x) + get_shannon_entropy(prob_z)

			# 4. Quantum Conditional Entropy
			s_ab = get_von_neumann_entropy(rho_ab)
			rho_b = np.trace(rho_ab.reshape(2, 2, 2, 2), axis1=0, axis2=2)
			s_b = get_von_neumann_entropy(rho_b)
			table_results.append([p, sum_h, s_ab - s_b])

		# --- Print Numerical Table ---
		print(f"{'p':<6} | {'H(X)+H(Z)':<12} | {'H(A|B)':<10}")
		print("-" * 35)
		for row in table_results:
			print(f"{row[0]:<6.2f} | {row[1]:<12.4f} | {row[2]:<10.4f}")

		# --- Visualization ---
		p_fine = np.linspace(0, 1, 100)
		h_sum_plot, h_cond_plot = [], []
		for p in p_fine:
			state = p * rho_pure + (1 - p) * np.eye(4) / 4
			ra = np.trace(state.reshape(2, 2, 2, 2), axis1=1, axis2=3)
			rb = np.trace(state.reshape(2, 2, 2, 2), axis1=0, axis2=2)
			hz = get_shannon_entropy(np.diag(ra).real)
			hx = get_shannon_entropy(np.diag(H_gate @ ra @ H_gate.T).real)
			h_sum_plot.append(hz + hx)
			h_cond_plot.append(get_von_neumann_entropy(state) - get_von_neumann_entropy(rb))

		plt.figure(figsize=(8, 5), dpi=150)
		plt.plot(p_fine, h_sum_plot, 'b-', label=r'$H(\sigma_x) + H(\sigma_z)$')
		plt.plot(p_fine, h_cond_plot, 'g--', label=r'$H(A|B)$')
		plt.axhline(y=1.0, color='r', linestyle=':', label='Classical Bound')
		plt.xlabel('Purity Parameter (p)')
		plt.ylabel('Entropy (Bits)')
		plt.legend()
		plt.grid(True, alpha=0.2)
		plt.show()