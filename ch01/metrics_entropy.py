
		import numpy as np
		import pandas as pd

		def binary_entropy(x):
			"""Calculates the binary von Neumann entropy."""
			if x <= 0 or x >= 1:
				return 0
			return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

		def analyze_metrics_rigorous(p_values):
			results = []

			for p in p_values:
				# 1. Fidelity for Werner state relative to |phi+>
				# F = <phi+| rho |phi+> = p + (1-p)/4 = (3p+1)/4
				fidelity = (3 * p + 1) / 4

				# 2. Concurrence (C) for Werner State
				# C = max(0, (3p-1)/2)
				concurrence = max(0, (3 * p - 1) / 2)

				# 3. Entanglement of Formation (EoF) / Cost (Ec)
				# Using Wootters formula: EoF = h( (1 + sqrt(1 - C^2))/2 )
				if concurrence > 0:
					val = (1 + np.sqrt(1 - concurrence**2)) / 2
					eof = binary_entropy(val)
				else:
					eof = 0.0

				# 4. Negativity (N)
				# N = max(0, (3p-1)/4)
				negativity = max(0, (3 * p - 1) / 4)

				results.append({
					'p': round(p, 2),
					'Fidelity': round(fidelity, 4),
					'Concurrence': round(concurrence, 4),
					'EoF/Cost': round(eof, 4),
					'Negativity': round(negativity, 4)
				})

			return pd.DataFrame(results)

		# Values of p for comparison (including thresholds p=0.33, 0.5, 0.9)
		p_test = [0.22, 0.33, 0.34, 0.5, 0.9, 1.0]
		df_comp = analyze_metrics_rigorous(p_test)

		print("Comparative Metrics Analysis (Rigorous):")
		print(df_comp.to_string(index=False))