		import numpy as np
		from scipy.linalg import eigvals
		
		def get_partial_trace_AB(rho_abc):
			"""
			Traces out the third qubit (C) correctly.
			Input: 8x8 matrix, Output: 4x4 matrix.
			"""
			# Reshape 8x8 matrix into a 6-index tensor: (a, b, c, a', b', c')
			tensor = rho_abc.reshape(2, 2, 2, 2, 2, 2)
			
			# Partial trace over the third qubit (indices c and c' are 2 and 5)
			# Remaining indices are a, b, a', b' (0, 1, 3, 4)
			rho_ab = np.einsum('abcadc->abcd', tensor).reshape(4, 4)
			return rho_ab
		
		def get_partial_transpose(rho_ab):
			"""Partial transpose on the second qubit of a 4x4 matrix."""
			tensor = rho_ab.reshape(2, 2, 2, 2)
			# Swap indices of the second qubit (indices 1 and 3) for partial transpose
			pt_tensor = np.einsum('iajb->ibja', tensor)
			return pt_tensor.reshape(4, 4)
		
		def get_negativity(rho_ab):
			"""Calculates negativity for a 4x4 matrix."""
			rho_pt = get_partial_transpose(rho_ab)
			evs = eigvals(rho_pt).real
			# Negativity: sum of absolute values of negative eigenvalues
			return np.sum((np.abs(evs) - evs) / 2)
		
		# 1. State vector construction
		ghz = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
		# W state: (|100> + |010> + |001>) / sqrt(3)
		w_state = np.zeros(8)
		w_state[[1, 2, 4]] = 1.0 / np.sqrt(3)
		
		# 2. Density matrix generation
		rho_ghz = np.outer(ghz, ghz)
		rho_w = np.outer(w_state, w_state)
		
		# 3. System reduction (Trace out qubit C)
		rho_ab_ghz = get_partial_trace_AB(rho_ghz)
		rho_ab_w = get_partial_trace_AB(rho_w)
		
		# 4. Numerical analysis
		neg_ghz = get_negativity(rho_ab_ghz)
		neg_w = get_negativity(rho_ab_w)
		
		# Calculate purity: Tr(rho^2)
		purity_ghz = np.real(np.trace(rho_ab_ghz @ rho_ab_ghz))
		purity_w = np.real(np.trace(rho_ab_w @ rho_ab_w))
		
		# 5. Output results
		print("-" * 55)
		print(f"{'State Class':<15} | {'Negativity (AB)':<18} | {'Purity'}")
		print("-" * 55)
		print(f"{'GHZ':<15} | {float(neg_ghz):<18.5f} | {float(purity_ghz):.4f}")
		print(f"{'W':<15} | {float(neg_w):<18.5f} | {float(purity_w):.4f}")
		print("-" * 55)