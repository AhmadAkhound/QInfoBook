
		import pandas as pd

		def calculate_hilbert_scaling(max_n):
			data = []
			for n in range(1, max_n + 1):
				dim = 2**n
				# Growth factor = d_n / d_{n-1}
				growth_factor = float(dim / (2**(n-1))) if n > 1 else 0.0
				data.append([n, dim, growth_factor])
			return data

		# Execution
		results = calculate_hilbert_scaling(10)
		df = pd.DataFrame(results, columns=['n', 'Dimension', 'Growth Factor'])
		print(df.to_string(index=False))