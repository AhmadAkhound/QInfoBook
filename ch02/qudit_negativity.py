		import numpy as np
		import pandas as pd

		def analyze_qudit_werner(d, p_values):
			results = []

			# 1. State Vector Construction
			phi_d = np.zeros((d**2, 1), dtype=complex)
			for i in range(d):
				vec_ii = np.zeros(d**2)
				vec_ii[i*d + i] = 1
				phi_d += vec_ii.reshape(-1, 1)
			phi_d = phi_d / np.sqrt(d)

			rho_phi = np.outer(phi_d, phi_d.conj())
			identity = np.eye(d**2) / (d**2)

			for p in p_values:
				# 2. Density Matrix Calculation
				rho = p * rho_phi + (1 - p) * identity

				# 3. Partial Transpose Operation
				rho_reshaped = rho.reshape(d, d, d, d)
				rho_pt = rho_reshaped.transpose(0, 3, 2, 1).reshape(d**2, d**2)

				# 4. Negativity Computation
				eigenvalues = np.linalg.eigvalsh(rho_pt)
				negative_evs = eigenvalues[eigenvalues < -1e-12]
				negativity = np.sum(np.abs(negative_evs))

				# 5. Logical Status Checking
				status = "NPT"
				if negativity < 1e-12:
					status = "PPT"

				# Checking critical point for d=3 (p=0.25)
				if abs(p - 0.25) < 0.001 and d == 3:
					status = "Critical"

				results.append([p, negativity, status])

			return pd.DataFrame(results, columns=['p', 'Negativity', 'Status'])

		# Execution
		d_val = 3
		p_list = [0.0, 0.2, 0.25, 0.3, 0.5, 1.0]
		df = analyze_qudit_werner(d_val, p_list)
		print(df.to_string(index=False))