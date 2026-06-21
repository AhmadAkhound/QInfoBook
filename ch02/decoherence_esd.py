import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_esd_analysis():
	# Set physical parameters
	gamma = 1.0  # Environmental decay rate
	t = np.linspace(0, 2, 100) # Dimensionless time
	
	# Modeling Amplitude Damping channel (each qubit independently)
	r = 1 - np.exp(-gamma * t)
	
	# Select initial X-state leading to ESD
	# Parameters: rho_11 = a, rho_44 = b, rho_23 = c
	a, b, c = 0.5, 0.1, 0.35 
	
	# Calculate Coherence (off-diagonal element rho_23)
	# Under amplitude damping applied independently to BOTH qubits,
	# each qubit contributes a factor sqrt(1-r) to this matrix element,
	# so the combined decay factor is (1-r) = exp(-gamma*t).
	coherence = c * (1 - r)
	
	# Calculate Negativity as entanglement measure
	# Neg = max(0, 2*(|rho_23| - sqrt(rho_11 * rho_44)))
	neg_raw = 2 * (c * (1-r) - np.sqrt(a * b * r**2))
	negativity = np.maximum(0, neg_raw)
	
	# Find exact ESD moment (where Negativity becomes zero)
	t_esd_idx = np.where(negativity == 0)[0][0]
	t_esd = t[t_esd_idx]
	
	# Extract data for LaTeX table (10 key points)
	indices = np.linspace(0, t_esd_idx + 15, 10, dtype=int) 
	indices = [i for i in indices if i < len(t)]
	
	table_data = pd.DataFrame({
		'Time (gamma*t)': t[indices],
		'Decay Rate (r)': r[indices],
		'Coherence': coherence[indices],
		'Negativity (N)': negativity[indices]
	})
	
	# System status based on Negativity value
	table_data['Status'] = table_data['Negativity (N)'].apply(
		lambda x: 'Entangled' if x > 0 else 'ESD Occurred'
	)
	
	# Plot high-quality comparative analysis
	plt.figure(figsize=(10, 6), dpi=150)
	plt.plot(t, coherence, 'b--', label='Coherence (Local)', linewidth=1.5)
	plt.plot(t, negativity, 'r-', label='Negativity (Entanglement)', linewidth=2.5)
	
	plt.axvline(x=t_esd, color='green', linestyle=':', label=f'ESD Moment (t={t_esd:.2f})')
	plt.fill_between(t, 0, negativity, color='red', alpha=0.1)
	
	plt.title("Comparative Analysis: Decoherence vs. Entanglement Sudden Death", fontsize=14)
	plt.xlabel(r"Dimensionless Time ($\gamma t$)", fontsize=12)
	plt.ylabel("Magnitude", fontsize=12)
	plt.legend()
	plt.grid(True, which='both', linestyle='--', alpha=0.5)
	
	print("--- Data for Your LaTeX Tables ---")
	print(table_data.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
	
	plt.show()

# Execute Simulation
generate_esd_analysis()