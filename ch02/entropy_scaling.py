		import numpy as np
		import matplotlib.pyplot as plt
		from scipy.interpolate import interp1d
		import os

		# Create figures directory for LaTeX integration
		if not os.path.exists('figures'):
			os.makedirs('figures')

		def calculate_entanglement_scaling(L, c=1.0):
			"""
			Calculates Entanglement Entropy based on CFT and Area Law physics.
			L: Total chain length
			c: Central charge
			"""
			l_values = np.arange(1, L)

			# 1. Critical Regime (CFT Scaling for finite systems)
			# S(l) = (c/3) * ln((L/pi) * sin(pi*l/L)) + const
			s_critical = (c/3) * np.log((L/np.pi) * np.sin(np.pi * l_values / L)) + 0.5

			# 2. Non-Critical Regime (Area Law Saturation)
			# S(l) approaches a constant value as l increases beyond correlation length xi
			xi = 1.2
			s_area_law = 0.8 * (1 - np.exp(-l_values / xi))

			return l_values, s_critical, s_area_law

		# System Parameters
		L_total = 10
		l_axis, s_crit, s_area = calculate_entanglement_scaling(L_total)

		# Plotting with high quality for book publication
		plt.figure(figsize=(8, 6), dpi=300)

		# Smooth interpolation for physical representation
		l_fine = np.linspace(1, L_total-1, 200)
		f_crit = interp1d(l_axis, s_crit, kind='cubic')
		f_area = interp1d(l_axis, s_area, kind='cubic')

		plt.plot(l_fine, f_crit(l_fine), color='red', linewidth=2.5, label='Critical (CFT)')
		plt.scatter(l_axis, s_crit, color='red', s=40, edgecolors='black', zorder=5)

		plt.plot(l_fine, f_area(l_fine), color='blue', linewidth=2.5, linestyle='--', label='Area Law')
		plt.scatter(l_axis, s_area, color='blue', s=40, edgecolors='black', zorder=5)

		plt.xlabel(r'Subsystem Size ($l$)')
		plt.ylabel(r'Entanglement Entropy $S(l)$')
		plt.title('Entanglement Entropy Scaling (L=10)')
		plt.grid(True, linestyle=':', alpha=0.7)
		plt.legend(loc='lower center')

		plt.tight_layout()
		plt.savefig('figures/entropy_scaling_plot.png')
		plt.show()

		# Table Data Export (Console output for Table 2.x)
		print(f"{'l':<4} | {'S_Critical':<12} | {'S_AreaLaw':<12}")
		print("-" * 35)
		for i in range(len(l_axis)):
			print(f"{l_axis[i]:<4} | {s_crit[i]:<12.4f} | {s_area[i]:<12.4f}")