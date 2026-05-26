
		import numpy as np
		import pandas as pd
		import matplotlib.pyplot as plt

		def binary_entropy(x):
			x = np.clip(x, 1e-15, 1 - 1e-15)
			return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

		def simulate_entropic_dynamics(p_values):
			data = []
			for p in p_values:
				fidelity = (3 * p + 1) / 4
				concurrence = max(0, (3 * p - 1) / 2)
				tangle = concurrence**2
				eof = 0.0
				if concurrence > 0:
					val = (1 + np.sqrt(1 - concurrence**2)) / 2
					eof = binary_entropy(val)
				data.append({'p': p, 'F': fidelity, 'T': tangle, 'EoF': eof})
			return pd.DataFrame(data)

		# Generating results for the table
		p_points = [0.0, 0.2, 0.33, 0.4, 0.6, 0.8, 1.0]
		df = simulate_entropic_dynamics(p_points)
		print(df.to_string(index=False))

		# Plotting
		p_fine = np.linspace(0, 1, 100)
		df_fine = simulate_entropic_dynamics(p_fine)
		plt.figure(figsize=(7, 5))
		plt.plot(p_fine, df_fine['F'], 'k--', label='Fidelity')
		plt.plot(p_fine, df_fine['EoF'], 'b-', label='EoF')
		plt.plot(p_fine, df_fine['T'], 'r-', label='Tangle')
		plt.axvline(x=1/3, color='gray', linestyle=':')
		plt.annotate('Entanglement Threshold', xy=(1/3, 0), xytext=(0.4, 0.1),
		arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
		plt.legend()
		plt.savefig('entropic_plot.png', dpi=300)