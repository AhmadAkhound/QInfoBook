
		import numpy as np
		import pandas as pd
		import matplotlib.pyplot as plt

		def analyze_werner_ppt(p_values):
			results = []
			# Bell singlet state |psi->
			singlet_psi = np.array([0, 1, -1, 0]) / np.sqrt(2)
			rho_singlet = np.outer(singlet_psi, singlet_psi)
			identity_matrix = np.eye(4) / 4

			for p in p_values:
				# Constructing Werner State: rho = p|psi><psi| + (1-p)/4 * I
				rho_werner = p * rho_singlet + (1 - p) * identity_matrix

				# Partial Transpose on subsystem B
				rho_reshaped = rho_werner.reshape(2, 2, 2, 2)
				rho_pt = rho_reshaped.transpose(0, 3, 2, 1).reshape(4, 4)

				# Extracting the smallest eigenvalue
				min_ev = np.min(np.linalg.eigvalsh(rho_pt))

				status = "Entangled (NPT)" if min_ev < -1e-12 else "Separable (PPT)"
				results.append([p, min_ev, status])

			return pd.DataFrame(results, columns=['p', 'Min Eigenvalue', 'Status'])

		# 1. Numerical Table Generation
		p_range = [0.0, 0.2, 1/3, 0.4, 0.6, 0.8, 1.0]
		output_df = analyze_werner_ppt(p_range)
		print(output_df.to_string(index=False))

		# 2. High-Quality Plot Generation for Figure 1.9
		p_fine = np.linspace(0, 1, 100)
		neg_vals = [max(0, (3*p - 1)/4) for p in p_fine]

		plt.figure(figsize=(8, 5), dpi=300)
		plt.plot(p_fine, neg_vals, 'b-', linewidth=2, label=r'Negativity $\mathcal{N}(\rho_W)$')
		plt.axvline(x=1/3, color='r', linestyle='--', label='Critical Limit (p=1/3)')
		plt.fill_between(p_fine, 0, neg_vals, where=(p_fine > 1/3), color='blue', alpha=0.1)
		plt.xlabel('Visibility Parameter (p)')
		plt.ylabel('Negativity')
		plt.title('Quantum Phase Transition in Werner State')
		plt.legend()
		plt.grid(alpha=0.3)
		plt.savefig('werner_final_plot.png', bbox_inches='tight')
		plt.show()