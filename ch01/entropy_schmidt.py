"""
========================================================================
Title       : Entanglement Entropy and Schmidt Rank
Book        : Quantum Entanglement, Volume 1
Chapter     : 1  --  Hilbert Space Foundations and Mathematical Tools
Code Ref    : entropy_schmidt
Description : Computes the von Neumann entropy of the reduced state and the
              Schmidt rank for a product state and a Bell state, linking the
              rank to the number of nonzero Schmidt coefficients.
========================================================================
"""
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
psi_m = np.array([[np.sqrt(3)/2],[0],[0],[0.5]])  # sqrt3/2|00> + 1/2|11>
states = [("Product", psi_p @ psi_p.T), ("Partial", psi_m @ psi_m.T), ("Bell", psi_b @ psi_b.T)]

print("Entropy & Schmidt Rank Analysis:")
for n, r in states:
	# Partial trace to get rho_A
	r_a = np.trace(r.reshape(2,2,2,2), axis1=1, axis2=3)
	ent = v_entropy(r_a); rank = get_schmidt_rank(r_a)
	print(f"State: {n:7} | Entropy: {ent:.2f} | Schmidt Rank: {rank}")