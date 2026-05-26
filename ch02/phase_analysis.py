		import numpy as np
		import pandas as pd

		def calculate_cv_entanglement_rigorous(r_values):
			"""
			Fundamental calculation of CV entanglement for TMSV.
			Based on Symplectic invariants of the Covariance Matrix.
			"""
			results = []

			for r in r_values:
				# 1. Physical Parameters (Standard Gaussian Form)
				# For TMSV, the elements of the covariance matrix are:
				# n = 1/2 * cosh(2r), c = 1/2 * sinh(2r)
				n = 0.5 * np.cosh(2 * r)
				c = 0.5 * np.sinh(2 * r)

				# 2. Covariance Matrix sigma = [[A, C], [C^T, B]]
				# A = B = n*I , C = c*diag(1, -1)
				A = np.array([[n, 0], [0, n]])
				B = np.array([[n, 0], [0, n]])
				C = np.array([[c, 0], [0, -c]])

				# Constructing the full 4x4 matrix
				sigma = np.block([[A, C], [C, B]])

				# 3. Partial Transpose (Mirror Inversion)
				# In phase space, PT(mode 2) means p2 -> -p2
				# This transforms C into C_tilde = diag(c, c)
				C_tilde = np.array([[c, 0], [0, c]])
				sigma_pt = np.block([[A, C_tilde], [C_tilde, B]])

				# 4. Calculation of Symplectic Eigenvalues of sigma_pt
				# For a 2-mode Gaussian state, we use the Delta invariant:
				# Delta = det(A) + det(B) + 2*det(C_tilde)
				det_A = np.linalg.det(A)
				det_B = np.linalg.det(B)
				det_C_tilde = np.linalg.det(C_tilde)
				det_sigma = np.linalg.det(sigma) # det(sigma) = det(sigma_pt)

				delta_tilde = det_A + det_B + 2 * det_C_tilde

				# Symplectic eigenvalue: nu = sqrt((Delta - sqrt(Delta^2 - 4*det))/2)
				inner_sqrt = max(0, delta_tilde**2 - 4 * det_sigma)
				nu_min = np.sqrt(0.5 * (delta_tilde - np.sqrt(inner_sqrt)))

				# 5. Logarithmic Negativity E_N = max(0, -log2(2 * nu_min))
				en = max(0, -np.log2(2 * nu_min)) if nu_min > 0 else 0

				results.append({
					'r': round(r, 2),
					'nu_min': round(nu_min, 4),
					'E_N': round(en, 4),
					'Status': 'Entangled' if nu_min < 0.4999 else 'Separable'
				})

			return pd.DataFrame(results)

		# Execution
		r_range = [0.0, 0.2, 0.5, 0.8, 1.0]
		df_results = calculate_cv_entanglement_rigorous(r_range)

		print("Rigorous Phase Space Analysis:")
		print(df_results.to_string(index=False))