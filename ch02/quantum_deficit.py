		import numpy as np
		import matplotlib.pyplot as plt

		def entropy(rho):
			"""Calculates Von Neumann Entropy."""
			evs = np.linalg.eigvalsh(rho)
			evs = evs[evs > 1e-12]
			return -np.sum(evs * np.log2(evs))

		def werner_state(p):
			"""Generates 2-qubit Werner state."""
			identity = np.eye(4) / 4
			psi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
			rho_bell = np.outer(psi_plus, psi_plus)
			return p * rho_bell + (1 - p) * identity

		def quantum_deficit(p):
			rho = werner_state(p)
			# 1. Total Information in the state (Log2(d) - S(rho))
			total_info = np.log2(4) - entropy(rho)

			# 2. Maximum information extractable locally
			# Representing the state after local dephasing (measurement in computational basis)
			rho_diag = np.diag(np.diag(rho))
			deficit = entropy(rho_diag) - entropy(rho)

			return total_info, max(0, deficit)

		# Execution and Data Generation
		p_values = np.linspace(0, 1, 15)
		results = [quantum_deficit(p) for p in p_values]

		# Formatting Table Output
		print(f"{'p (Purity)':<12} | {'Total Info':<12} | {'Quantum Deficit':<15}")
		print("-" * 45)
		for p, (info, defic) in zip(p_values, results):
			print(f"{p:<12.4f} | {info:<12.4f} | {defic:<15.4f}")

		# Plotting for high-quality output
		p_fine = np.linspace(0, 1, 50)
		defic_fine = [quantum_deficit(p)[1] for p in p_fine]

		plt.figure(figsize=(8, 5), dpi=150)
		plt.plot(p_fine, defic_fine, 'g-', lw=2, label='Quantum Deficit')
		plt.axvline(x=1/3, color='gray', linestyle=':', label='Entanglement Bound (p=1/3)')
		plt.xlabel('Purity parameter (p)')
		plt.ylabel('Deficit (Bits)')
		plt.title('Quantum Deficit in Werner State')
		plt.legend()
		plt.grid(alpha=0.3)
		plt.show()