		import numpy as np
		import matplotlib.pyplot as plt

		def concurrence(rho):
			"""Calculates Concurrence for a 2-qubit state."""
			sigma_y = np.array([[0, -1j], [1j, 0]])
			rho_tilde = np.kron(sigma_y, sigma_y) @ rho.conj() @ np.kron(sigma_y, sigma_y)
			R = rho @ rho_tilde
			evs = np.linalg.eigvals(R)
			evs = np.sort(np.real(np.sqrt(evs + 1e-15)))[::-1]
			return max(0, evs[0] - evs[1] - evs[2] - evs[3])

		def calculate_robustness(p_steps=20):
			# Base entangled state (Bell state)
			psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
			rho_pure = np.outer(psi, psi)

			# White noise state (Identity / 4)
			noise = np.eye(4) / 4

			p_values = np.linspace(0, 1, p_steps)
			conc_values = []

			for p in p_values:
				# Mixed state: rho = (1-p)*rho_pure + p*noise
				rho_mixed = (1 - p) * rho_pure + p * noise
				conc_values.append(concurrence(rho_mixed))

			return p_values, conc_values

		# Execution
		p_vals, c_vals = calculate_robustness()

		# Numerical Output for Table
		print(f"{'Noise (p)':<12} | {'Concurrence':<15}")
		print("-" * 30)
		for p, c in zip(p_vals, c_vals):
			print(f"{p:<12.4f} | {c:<15.4f}")

		# Plotting for visualization
		plt.figure(figsize=(8, 5), dpi=150)
		plt.plot(p_vals, c_vals, 'm-o', lw=2, label='Concurrence decay')
		plt.axvline(x=2/3, color='red', linestyle='--', label='Robustness Limit (p=2/3)')
		plt.fill_between(p_vals, c_vals, color='magenta', alpha=0.1)
		plt.xlabel('Noise Parameter (p)')
		plt.ylabel('Concurrence')
		plt.title('Robustness Analysis: Transition to Separability')
		plt.legend()
		plt.grid(alpha=0.3)
		plt.show()