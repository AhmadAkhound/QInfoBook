		import numpy as np
		import matplotlib.pyplot as plt

		def is_entangled_by_witness(rho, witness):
			"""
			Calculates the expectation value Tr(W * rho).
			A negative result indicates confirmed entanglement.
			"""
			# Calculation of the Trace of the product of two matrices
			return np.trace(witness @ rho).real

		# 1. Physical Setup: Constructing the Bell State |Phi+>
		# |Phi+> = 1/sqrt(2) * (|00> + |11>)
		phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
		P_phi_plus = np.outer(phi_plus, phi_plus)  # Projector |Phi+><Phi+|

		# 2. Witness Design: W = 0.5 * I - |Phi+><Phi+|
		# This is an optimal witness for the Bell state Phi+
		identity_4 = np.eye(4)
		W_bell = 0.5 * identity_4 - P_phi_plus

		# 3. Werner State Simulation over Purity Parameter p
		p_values = np.linspace(0, 1, 100)
		witness_vals = []

		for p in p_values:
			# Rho_Werner = p*|Phi+><Phi+| + (1-p)/4 * I
			rho_werner = p * P_phi_plus + (1 - p) / 4 * identity_4
			val = is_entangled_by_witness(rho_werner, W_bell)
			witness_vals.append(val)

		# 4. Numerical Table Generation for Exercise Verification
		print(f"{'p':<6} | {'<W>':<10} | {'Detection Status'}")
		print("-" * 40)
		check_points = [0.0, 0.2, 0.33, 0.5, 0.8, 1.0]
		for p in check_points:
			rho = p * P_phi_plus + (1 - p) / 4 * identity_4
			val = is_entangled_by_witness(rho, W_bell)
			status = "ENTANGLED (Negative)" if val < 0 else "SEPARABLE / UNDETECTED"
			print(f"{p:<6.2f} | {val:<10.4f} | {status}")

		# 5. High-Quality Visualization
		plt.figure(figsize=(8, 5), dpi=150)
		plt.plot(p_values, witness_vals, 'b-', linewidth=2, label=r'Expectation Value $\langle W \rangle$')
		plt.axhline(y=0, color='r', linestyle='--', label='Classical/Quantum Boundary')
		plt.axvline(x=1/3, color='g', linestyle=':', label='Theoretical Limit ($p=1/3$)')

		# Highlight the detection region where Trace(W*rho) < 0
		plt.fill_between(p_values, witness_vals, 0, where=(np.array(witness_vals) < 0),
						color='red', alpha=0.15, label='Entanglement Detected')

		plt.xlabel('Purity Parameter (p)')
		plt.ylabel(r'Witness Expectation Value $\langle W \rangle$')
		plt.title('Entanglement Witness Performance on Werner States')
		plt.legend(loc='upper right')
		plt.grid(True, alpha=0.2)
		plt.tight_layout()
		plt.show()