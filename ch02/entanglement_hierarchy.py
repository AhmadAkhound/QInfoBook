
		import numpy as np
		import matplotlib.pyplot as plt

		def binary_entropy(x):
			"""Calculates the binary Shannon entropy."""
			if x <= 0 or x >= 1:
				return 0
			return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

		def calculate_measures(p):
			# 1. Concurrence & Tangle (C^2)
			concurrence = max(0, (3 * p - 1) / 2)
			tangle = concurrence**2

			# 2. Entanglement of Formation (EoF)
			if concurrence == 0:
				eof = 0.0
			else:
				val = (1 + np.sqrt(1 - concurrence**2)) / 2
				eof = binary_entropy(val)

			# 3. Relative Entropy of Entanglement (REE)
			if p <= 1/3:
				ree = 0.0
			elif p >= 0.999: # Handling the pure state limit numerically
				ree = 1.0
			else:
				# Based on the closest separable state at the boundary p=1/3
				rho_vals = [(1+3*p)/4, (1-p)/4, (1-p)/4, (1-p)/4]
				sig_vals = [1/2, 1/6, 1/6, 1/6]
				ree = sum(r * np.log2(r/s) for r, s in zip(rho_vals, sig_vals) if r > 0)

			return tangle, eof, ree

		# --- Data Generation ---
		p_values = np.array([0.0, 0.33, 0.4, 0.6, 0.8, 1.0])
		p_fine = np.linspace(0, 1, 200)

		print(f"{'p':<6} | {'Tangle':<10} | {'EoF':<10} | {'REE':<10}")
		print("-" * 45)

		for p in p_values:
			t, e, r = calculate_measures(p)
			print(f"{p:<6.2f} | {t:<10.4f} | {e:<10.4f} | {r:<10.4f}")

		# --- Plotting ---
		t_plot, e_plot, r_plot = [], [], []
		for p in p_fine:
			t, e, r = calculate_measures(p)
			t_plot.append(t)
			e_plot.append(e)
			r_plot.append(r)

		plt.figure(figsize=(9, 6), dpi=150)
		plt.plot(p_fine, t_plot, 'k:', label=r'Tangle ($C^2$)')
		plt.plot(p_fine, e_plot, 'b-', label='Entanglement of Formation')
		plt.plot(p_fine, r_plot, 'r--', label='Relative Entropy of Entanglement')

		plt.axvline(x=1/3, color='gray', linestyle='-', alpha=0.3)
		plt.xlabel('Purity Parameter (p)')
		plt.ylabel('Measure Value (Bits)')
		plt.title('Hierarchy of Quantum Entanglement Measures')
		plt.legend()
		plt.grid(True, alpha=0.2)
		plt.show()