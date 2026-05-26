				import numpy as np

				def calculate_eof_simulation():
					"""
					Simulating the Entropy of Formation (EoF) for a mixture:
					rho = p * |Phi+><Phi+| + (1-p) * |00><00|
					"""
					# Defining the parameter p (Weight of Entanglement)
					p_steps = np.linspace(0, 1, 6)

					# Defining Basis States
					# Pure Bell State |Phi+> = 1/sqrt(2) * (|00> + |11>)
					phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
					rho_ent = np.outer(phi_plus, phi_plus.conj())

					# Separable State |00>
					phi_sep = np.array([1, 0, 0, 0])
					rho_sep = np.outer(phi_sep, phi_sep.conj())

					# Storage for results
					results = []

					for p in p_steps:
						# Constructing the mixed state density matrix
						rho_mixed = p * rho_ent + (1 - p) * rho_sep

						# In this specific decomposition, the EoF bound is:
						# Ef <= p * S(rho_ent_reduced) + (1-p) * S(rho_sep_reduced)
						ef_upper_bound = p * 1.0 + (1 - p) * 0.0

						results.append((p, ef_upper_bound))

					return results

				# Execution and Data Presentation
				sim_data = calculate_eof_simulation()

				print(f"{'Weight (p)':<12} | {'EoF Bound':<12}")
				print("-" * 28)
				for p, ef in sim_data:
					print(f"{p:<12.2f} | {ef:<12.4f}")

				if __name__ == "__main__":
					calculate_eof_simulation()