
		import numpy as np
		import pandas as pd

		def binary_entropy(x):
			"""Calculates the binary von Neumann entropy: H(x) = -x log2(x) - (1-x) log2(1-x)"""
			if x <= 0 or x >= 1:
				return 0
			return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

		def calculate_eof_werner(p_values):
			"""Calculates EoF for a 2-qubit Werner state using the analytical formula."""
			results = []
			for p in p_values:
				# 1. Calculation of Concurrence (C) for Werner State: C = max(0, (3p-1)/2)
				concurrence = max(0, (3 * p - 1) / 2)

				# 2. Calculation of EoF from Concurrence:
				# E_f = H( (1 + sqrt(1 - C^2)) / 2 )
				if concurrence > 0:
					c_square = concurrence**2
					val = (1 + np.sqrt(1 - c_square)) / 2
					eof = binary_entropy(val)
				else:
					eof = 0.0

				results.append({
					'p (Purity)': round(p, 3),
					'Concurrence': round(concurrence, 4),
					'EoF': round(eof, 4)
				})
			return pd.DataFrame(results)

		# Points requested in the textbook (including p=0.5 and the threshold 1/3)
		p_points = [0.0, 0.333, 0.334, 0.5, 0.75, 1.0]
		df_eof = calculate_eof_werner(p_points)

		print("Entanglement of Formation Results:")
		print(df_eof.to_string(index=False))