		import numpy as np
		import matplotlib.pyplot as plt

		def shannon_entropy(lambdas):
			"""Calculates Von Neumann entropy for a given Schmidt spectrum."""
			l = np.array(lambdas)
			l = l[l > 0]
			return -np.sum(l * np.log2(l))

		def check_locc_possibility(l_init, l_target):
			"""Nielsen's Theorem: psi -> phi iff lambda_psi is majorized by lambda_phi."""
			s_init = np.sort(l_init)[::-1]
			s_target = np.sort(l_target)[::-1]
			# Core majorization condition: sum(init) <= sum(target) for all k
			return np.all(np.cumsum(s_init) <= np.cumsum(s_target) + 1e-12)

		# --- 1. Generate Table Data and Analyze Transitions ---
		test_cases = [
			([0.8, 0.2], [0.6, 0.4], "Try to increase entanglement"),
			([0.5, 0.5], [0.9, 0.1], "Convert Maximal to Partial"),
			([0.7, 0.3], [0.7, 0.3], "Identity map"),
			([1.0, 0.0], [0.5, 0.5], "Create from product state")
		]

		print(f"{'Row':<4} | {'Initial (L)':<12} | {'Target (L)':<12} | {'Possible?':<10} | {'Entropy Change'}")
		print("-" * 65)
		for i, (init, target, desc) in enumerate(test_cases, 1):
			possible = check_locc_possibility(init, target)
			h_i, h_t = shannon_entropy(init), shannon_entropy(target)
			status = "Yes" if possible else "No"
			print(f"{i:<4} | {str(init):<12} | {str(target):<12} | {status:<10} | {h_i:.2f} -> {h_t:.2f}")

		# --- 2. Generate Distillation Plot (Figure 28) ---
		plt.figure(figsize=(8, 5), dpi=150)
		x = np.linspace(0, 10, 100)
		# Conceptual curve: Fidelity saturating towards 1 (Bell State)
		y = 1 - 0.5 * np.exp(-0.4 * x)

		plt.plot(x, y, 'b-', linewidth=2.5, label='Output Fidelity ($F$)')
		plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Bell State ($F=1$)')
		plt.axhline(y=0.5, color='k', linestyle=':', label='Classical Limit')

		plt.fill_between(x, y, 0.5, color='blue', alpha=0.1)
		plt.title("Entanglement Concentration/Distillation Dynamics", fontsize=12)
		plt.xlabel("Number of Iterations / Copies Processed", fontsize=10)
		plt.ylabel("Fidelity of the Target Pair", fontsize=10)
		plt.grid(True, linestyle=':', alpha=0.6)
		plt.legend(loc='lower right')
		plt.tight_layout()
		plt.savefig("distillation_plot.png")
		plt.show()