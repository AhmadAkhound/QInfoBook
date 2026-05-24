
		import numpy as np
		def get_purity(rho): return np.real(np.trace(rho @ rho))
		def get_pt_A(r_AB):
			basis=[np.array([[1],[0]]),np.array([[0],[1]])]
			ops=[np.kron(np.eye(2),b.T) for b in basis]
			return sum([o @ r_AB @ o.T for o in ops])

		rho_prod=np.zeros((4,4)); rho_prod[0,0]=1.0
		psi_b=np.array([[0],[1],[-1],[0]])/np.sqrt(2); rho_bell=psi_b @ psi_b.T
		states=[("Product",rho_prod),("Bell",rho_bell)]
		print("Results:")
		for n,r in states:
			p_t=get_purity(r); r_a=get_pt_A(r); p_s=get_purity(r_a)
			print(f"State: {n:7} | Total: {p_t:.1f} | Sub: {p_s:.1f}")