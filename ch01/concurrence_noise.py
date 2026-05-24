
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.linalg import eigvals

	def calculate_concurrence(rho):
		# Ab initio calculation: spin-flip transformation and eigenvalues
		sy = np.array([[0, -1j], [1j, 0]])
		sy2 = np.kron(sy, sy)
		rho_tilde = sy2 @ rho.conj() @ sy2
		R = rho @ rho_tilde
		evals = eigvals(R)
		# Sorting roots of eigenvalues in descending order
		lams = np.sort(np.sqrt(np.maximum(evals.real, 0)))[::-1]
		return max(0, lams[0] - np.sum(lams[1:]))

	# Simulation settings
	gamma_range = np.linspace(0, 1, 6)
	c_unitary_list = []
	c_noise_list = []

	# Initial Bell state: |Phi+> (Maximally Entangled)
	phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
	rho_0 = np.outer(phi_plus, phi_plus.conj())

	# Local Unitary Operator (X-gate on the first qubit)
	u_local = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2))

	for g in gamma_range:
		# 1. Local Unitary Evolution: Concurrence must remain invariant (C=1)
		rho_u = u_local @ rho_0 @ u_local.conj().T
		c_unitary_list.append(calculate_concurrence(rho_u))

		# 2. Noisy Evolution (Depolarizing noise model): Monotonic decay
		rho_g = (1 - g) * rho_0 + g * np.eye(4) / 4
		c_noise_list.append(calculate_concurrence(rho_g))

	# Numerical Table Output
	print(f"{'Gamma':<10} | {'C_Unitary':<12} | {'C_Noise':<10}")
	print("-" * 40)
	for g, cu, cn in zip(gamma_range, c_unitary_list, c_noise_list):
		print(f"{g:<10.1f} | {cu:<12.4f} | {cn:<10.4f}")

	# Quality Plotting
	plt.figure(figsize=(7, 5), dpi=100)
	plt.plot(gamma_range, c_unitary_list, 'r--', linewidth=2, label='Local Unitary (Invariance)')
	plt.plot(gamma_range, c_noise_list, 'b-o', markersize=6, label='Noisy Channel (Monotonicity)')
	plt.xlabel('Noise Parameter (gamma)')
	plt.ylabel('Concurrence (C)')
	plt.title('Physical Validation: Invariance vs. Monotonicity')
	plt.legend()
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.show()