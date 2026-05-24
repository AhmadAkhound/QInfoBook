
		import numpy as np
		def v_entropy(rho):
			evs=np.linalg.eigvalsh(rho); evs=evs[evs>1e-12]
			return -np.sum(evs*np.log2(evs))

		def get_schmidt_rank(rho):
			evs=np.linalg.eigvalsh(rho)
			return np.count_nonzero(evs > 1e-12)

		# Define States
		psi_p = np.array([[1],[0],[0],[0]]) # Product |00>
		psi_b = np.array([[1],[0],[0],[1]])/np.sqrt(2) # Bell |Phi+>
		states = [("Product", psi_p @ psi_p.T), ("Bell", psi_b @ psi_b.T)]

		print("Entropy & Schmidt Rank Analysis:")
		for n, r in states:
			# Partial trace to get rho_A
			r_a = np.trace(r.reshape(2,2,2,2), axis1=1, axis2=3)
			ent = v_entropy(r_a); rank = get_schmidt_rank(r_a)
			print(f"State: {n:7} | Entropy: {ent:.2f} | Schmidt Rank: {rank}")