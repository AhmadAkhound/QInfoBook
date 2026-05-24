basicstyle=\ttfamily\color{black}, stringstyle=\color{black}]
		import numpy as np
		import matplotlib.pyplot as plt
		import pandas as pd

		def von_neumann_entropy(rho):
			"""Calculates Von Neumann Entropy of a density matrix."""
			evals = np.linalg.eigvalsh(rho)
			evals = evals[evals > 1e-12]  # Numerical stability for log calculation
			return -np.sum(evals * np.log2(evals))

		def simulate_entropy_sweep():
			# Defining the theta range from 0 to 90 degrees
			thetas = np.linspace(0, np.pi/2, 50)
			results = []

			for t in thetas:
				# State: cos(t)|00> + sin(t)|11>
				c, s = np.cos(t), np.sin(t)
				# Reduced density matrix rho_A after partial trace
				rho_a = np.array([[c**2, 0], [0, s**2]])

				entropy = von_neumann_entropy(rho_a)
				results.append([np.degrees(t), c**2, entropy])

			return np.array(results)

		# Execution and Data Generation
		data = simulate_entropy_sweep()
		df = pd.DataFrame(data, columns=['Theta (deg)', 'cos^2(theta)', 'Entropy S(rho_A)'])

		# High-Quality Visualization
		plt.figure(figsize=(8, 5), dpi=150)
		plt.plot(data[:, 0], data[:, 2], 'b-', linewidth=2, label=r'$S(\rho_A)$')
		plt.axvline(x=45, color='r', linestyle='--', label='Max Entanglement (Bell State)')
		plt.title('Entanglement Entropy vs. State Parameters', fontsize=12)
		plt.xlabel(r'$\theta$ (Degrees)', fontsize=11)
		plt.ylabel('Von Neumann Entropy', fontsize=11)
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.show()

		# Display results for specific points (as shown in Table 1.x)
		print("--- Numerical Output for Table Construction ---")
		print(df.iloc[::10].to_string(index=False))