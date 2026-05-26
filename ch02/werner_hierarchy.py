
		import numpy as np
		import pandas as pd
		import matplotlib.pyplot as plt

		def calculate_werner_metrics(p_values):
			results = []
			for p in p_values:
				# 1. Fidelity: F = (3p + 1) / 4
				fidelity = (3 * p + 1) / 4

				# 2. Concurrence: C = max(0, (3p - 1) / 2)
				concurrence = max(0, (3 * p - 1) / 2)

				# 3. Negativity: N = max(0, (3p - 1) / 4)
				negativity = max(0, (3 * p - 1) / 4)

				# 4. Ratio C/N (only if N > 0)
				ratio = concurrence / negativity if negativity > 0 else np.nan

				# 5. Entanglement of Formation (EoF)
				eof = 0.0
				if concurrence > 0:
					c2 = concurrence**2
					val = (1 + np.sqrt(1 - c2)) / 2
					# Binary entropy function
					eof = -val * np.log2(val) - (1 - val) * np.log2(1 - val) if val < 1 else 1.0

				results.append({
					'p': p,
					'Concurrence': concurrence,
					'Negativity': negativity,
					'Fidelity': fidelity,
					'EoF': eof,
					'C/N_Ratio': ratio
				})
			return pd.DataFrame(results)

		# Points matching Table 2.3 exactly
		p_points = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		df = calculate_werner_metrics(p_points)

		print("Numerical Data for Table 2.3 & Exercise 2.2:")
		print(df[['p', 'Concurrence', 'Negativity', 'Fidelity', 'C/N_Ratio']].to_string(index=False, float_format="%.4f"))

		# Plotting for Figure 2.3 and 2.4
		p_fine = np.linspace(0, 1, 300)
		df_fine = calculate_werner_metrics(p_fine)

		plt.figure(figsize=(9, 6), dpi=100)
		plt.plot(p_fine, df_fine['Fidelity'], 'k--', label='Fidelity (F)', alpha=0.6)
		plt.plot(p_fine, df_fine['Concurrence'], 'b-', label='Concurrence (C)', linewidth=2)
		plt.plot(p_fine, df_fine['Negativity'], 'g-.', label='Negativity (N)', linewidth=2)
		plt.plot(p_fine, df_fine['EoF'], 'r:', label='EoF (Non-linear)', linewidth=2.5)

		plt.axvline(x=1/3, color='orange', linestyle='-', label='Phase Transition (p=1/3)')
		plt.fill_between(p_fine, 0, 1, where=(p_fine <= 1/3), color='gray', alpha=0.1, label='Separable Region')

		plt.title('Hierarchy of Entanglement Measures in Werner States')
		plt.xlabel('Purity Parameter (p)')
		plt.ylabel('Measure Value')
		plt.legend(loc='upper left')
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()