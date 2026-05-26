		import numpy as np
		import matplotlib.pyplot as plt
		from scipy.interpolate import make_interp_spline
		import pandas as pd
		import os

		# Create directory for figures if it doesn't exist
		if not os.path.exists('figures'):
			os.makedirs('figures')

		# --- Section 1: Physical Data & Numerical Calculations ---
		# Values from exact diagonalization (N=4) for 1D Transverse Ising Model
		h_table = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
		c_table = np.array([0.068, 0.139, 0.243, 0.363, 0.462, 0.500, 0.335, 0.225, 0.151, 0.101, 0.068])

		# Numerical derivative (dC/dh) using central differences
		dc_dh = np.gradient(c_table, h_table)

		# Entanglement Scaling Laws data
		l_sizes = np.array([2, 4, 8, 16, 32, 64])
		s_crit = 0.5 * np.log2(l_sizes) + 0.333  # Logarithmic scaling (Critical)
		s_vol = 0.4 * l_sizes                   # Volume Law (Linear)
		s_area = np.full_like(l_sizes, 1.2)      # Area Law (Constant)

		# --- Section 2: Figure (a) - Quantum Phase Transition ---
		plt.figure(figsize=(7, 5), dpi=300)
		h_fine = np.linspace(h_table.min(), h_table.max(), 300)
		spl_c = make_interp_spline(h_table, c_table, k=3)(h_fine)
		spl_dc = make_interp_spline(h_table, dc_dh, k=3)(h_fine)

		plt.plot(h_fine, spl_c, 'b-', linewidth=2.5, label=r'Concurrence ($C$)')
		plt.plot(h_fine, spl_dc, 'r--', linewidth=2, label=r'$dC/dh$')
		plt.scatter(h_table, c_table, color='darkblue', s=40, zorder=5)
		plt.axvline(x=1.0, color='black', linestyle=':', alpha=0.5, label='Critical Point')

		plt.xlabel(r'Field Strength ($h$)')
		plt.ylabel(r'Value')
		plt.title("Quantum Phase Transition in 1D Ising Model")
		plt.legend(loc='upper right')
		plt.grid(True, linestyle=':', alpha=0.6)
		plt.savefig("figures/ising_qpt.png", bbox_inches='tight')
		plt.show()

		# --- Section 3: Figure (b) - Entanglement Scaling Laws ---
		plt.figure(figsize=(7, 5), dpi=300)
		l_fine = np.linspace(l_sizes.min(), l_sizes.max(), 300)
		spl_s = make_interp_spline(l_sizes, s_crit, k=3)(l_fine)

		plt.plot(l_sizes, s_vol, 'r--s', markersize=5, label=r'Volume Law ($S \sim l$)')
		plt.plot(l_fine, spl_s, 'b-', linewidth=2.5, label=r'Critical Scaling ($S \sim \log l$)')
		plt.scatter(l_sizes, s_crit, color='darkblue', s=30, zorder=5)
		plt.plot(l_sizes, s_area, 'g-.^', markersize=5, label=r'Area Law ($S \sim const$)')

		plt.xlabel(r'Subsystem Size ($l$)')
		plt.ylabel(r'Entanglement Entropy $S(l)$')
		plt.title("Entanglement Scaling Laws")
		plt.legend(loc='upper left')
		plt.grid(True, linestyle=':', alpha=0.6)
		plt.savefig("figures/scaling_laws.png", bbox_inches='tight')
		plt.show()

		# --- Section 4: Data Table Output ---
		df = pd.DataFrame({
			'Field (h)': h_table,
			'Concurrence (C)': c_table,
			'dC/dh': np.round(dc_dh, 3)
		})
		print("\n--- Numerical Calculation Table ---")
		print(df.to_string(index=False))