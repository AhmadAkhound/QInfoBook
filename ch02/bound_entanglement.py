		import numpy as np
		from numpy import linalg as LA

		def partial_transpose(rho, dims, subsystem=1):
			d1, d2 = dims
			reshaped = rho.reshape([d1, d2, d1, d2])
			if subsystem == 1:
				transposed = reshaped.transpose([2, 1, 0, 3])
			else:
				transposed = reshaped.transpose([0, 3, 2, 1])
			return transposed.reshape([d1*d2, d1*d2])

		# Constructing the 3x3 Tiles State based on UPB
		e0, e1, e2 = np.eye(3)
		upb = [
			np.outer(e0, (e0-e1)/np.sqrt(2)),
			np.outer((e0-e1)/np.sqrt(2), e2),
			np.outer(e2, (e1-e2)/np.sqrt(2)),
			np.outer((e1-e2)/np.sqrt(2), e0),
			np.outer((e0+e1+e2)/np.sqrt(3), (e0+e1+e2)/np.sqrt(3))
		]

		projection_upb = sum(np.outer(p.flatten(), np.conj(p.flatten())) for p in upb)
		rho_bound = (np.eye(9) - projection_upb) / 4.0

		# Spectrum Analysis
		eig_rho = LA.eigvalsh(rho_bound)
		eig_pt = LA.eigvalsh(partial_transpose(rho_bound, (3, 3)))

		print(f"{'Index':<5} | {'Eig_Rho':<10} | {'Eig_PT':<10}")
		for i in range(9):
			print(f"{i+1:<5} | {eig_rho[i]:<10.4f} | {eig_pt[i]:<10.4f}")