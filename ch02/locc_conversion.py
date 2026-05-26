		import numpy as np

		def check_majorization(lambda_psi, lambda_phi):
			"""
			Checks if lambda_psi is majorized by lambda_phi (psi < phi).
			According to Nielsen's theorem, this means psi -> phi is possible via LOCC.
			"""
			# 1. Sort eigenvalues in descending order for both states
			s_psi = np.sort(lambda_psi)[::-1]
			s_phi = np.sort(lambda_phi)[::-1]

			# 2. Check for dimension consistency
			if len(s_psi) != len(s_phi):
				return "Dimension Mismatch", None, None

			# 3. Calculate cumulative sums (Partial Sums)
			sums_psi = np.cumsum(s_psi)
			sums_phi = np.cumsum(s_phi)

			# 4. Verification of majorization conditions:
			# Sum_{i=1}^k psi_i <= Sum_{i=1}^k phi_i for all k
			is_majorized = True
			for i in range(len(sums_psi)):
				if sums_psi[i] > sums_phi[i] + 1e-10:  # Tolerance for float precision
					is_majorized = False
					break

			return is_majorized, sums_psi, sums_phi

		def shannon_entropy(lambdas):
			"""Calculates Shannon Entropy: H = -Sum(p * log2(p))"""
			nonzero_l = lambdas[lambdas > 0]
			return -np.sum(nonzero_l * np.log2(nonzero_l))

		# Define state spectra for testing
		# Example 1: Standard qubit case (From textbook example)
		psi_a = np.array([0.8, 0.2])
		phi_a = np.array([0.6, 0.4])

		# Example 2: Incomparable states in d=4 (Neither majorizes the other)
		psi_b = np.array([0.5, 0.25, 0.25, 0.0])
		phi_b = np.array([0.4, 0.4, 0.1, 0.1])

		test_cases = [("Text Example", psi_a, phi_a),
					("Incomparable Dim-4", psi_b, phi_b)]

		print(f"{'Test Case':<20} | {'Status (psi -> phi)':<22} | {'Entropy Ratio'}")
		print("-" * 70)

		for name, p, f in test_cases:
			possible, _, _ = check_majorization(p, f)
			h_p = shannon_entropy(p)
			h_f = shannon_entropy(f)

			res = "Possible" if possible else "Impossible"
			print(f"{name:<20} | {res:<22} | {h_p:.3f} vs {h_f:.3f} bits")