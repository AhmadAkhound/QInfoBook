		import numpy as np
		import matplotlib.pyplot as plt

		def witness_expectation(p):
			"""Theoretical expectation value for Werner states: (1 - 3p) / 4"""
			return (1 - 3 * p) / 4

		# 1. High-resolution simulation data
		p_fine = np.linspace(0, 1, 1000)
		w_fine = witness_expectation(p_fine)

		# 2. Extracting the Numerical Critical Point
		# Finding where the sign changes from positive to negative
		crossing_indices = np.where(np.diff(np.sign(w_fine)))[0]
		p_crit_num = p_fine[crossing_indices[0]]

		# 3. Data for Table 2.x (0.0 to 1.0 with 0.1 increments)
		p_steps = np.linspace(0, 1, 11)
		w_steps = witness_expectation(p_steps)

		print(f"{'p':<5} | {'<W>':<8} | {'Status'}")
		print("-" * 30)
		for p, w in zip(p_steps, w_steps):
			status = "DETECTED" if w < 0 else "UNDETECTED"
			print(f"{p:<5.1f} | {w:<8.3f} | {status}")

		print(f"\nNumerical Critical Point: p = {p_crit_num:.4f}")
		print(f"Theoretical Critical Point: p = {1/3:.4f}")

		# 4. Visualization for Figure generation
		plt.figure(figsize=(8, 5), dpi=150)

		# Plotting the main witness line
		plt.plot(p_fine, w_fine, color='#1f77b4', linewidth=2, label=r'$\langle W \rangle$ (Simulated)')

		# Shading the Entanglement Detection Zone (Negative Region)
		plt.fill_between(p_fine, w_fine, 0, where=(w_fine < 0),
		color='green', alpha=0.2, hatch='//', label='Detection Zone')

		# Reference Lines
		plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
		plt.axvline(1/3, color='red', linestyle='--', alpha=0.6, label='Theoretical Limit (1/3)')

		# Annotation of the Crossing Point
		plt.scatter([p_crit_num], [0], color='red', zorder=5)
		plt.annotate(f'Crossing: p={p_crit_num:.3f}', xy=(p_crit_num, 0), xytext=(0.4, 0.1),
					arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

		plt.xlabel('Purity Parameter (p)')
		plt.ylabel(r'Expectation Value $\langle W \rangle$')
		plt.title('Transition from Separable to Entangled Region')
		plt.grid(True, linestyle=':', alpha=0.5)
		plt.legend()
		plt.tight_layout()
		plt.show()