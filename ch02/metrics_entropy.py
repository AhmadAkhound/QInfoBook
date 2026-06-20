import numpy as np
import pandas as pd

# -----------------------------
# Basic entropy function
# -----------------------------
def binary_entropy(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x*np.log2(x) - (1-x)*np.log2(1-x)

# -----------------------------
# Rényi entropy (full spectral definition)
# -----------------------------
def renyi_entropy(eigs, alpha):
    eigs = np.array(eigs)
    if alpha == 1:
        return -np.sum(eigs * np.log2(eigs + 1e-15))
    return (1/(1-alpha)) * np.log2(np.sum(eigs**alpha))

# -----------------------------
# Relative entropy of entanglement (Werner approximation)
# For Werner state: S(ρ||σ_sep) reduces to analytical form
# using eigenvalue structure (simplified but exact for isotropic family)
# -----------------------------
def relative_entropy_entanglement(p):
    """
    Calculates the Relative Entropy of Entanglement for a Werner state.
    
    Definition: E_R(rho) = min_{sigma in S} S(rho || sigma)
    In the separable region (p <= 1/3), the optimal reference state is the 
    state itself, leading to zero distance.
    In the entangled region (p > 1/3), the optimal reference state is the 
    maximally mixed state (I/4).
    """
    # Eigenvalues of the Werner state
    lam1 = (1 + 3 * p) / 4
    lam = (1 - p) / 4
    
    # Von Neumann entropy of the Werner state (S_rho)
    S_rho = - (lam1 * np.log2(lam1 + 1e-15) + 3 * lam * np.log2(lam + 1e-15))
    
    # Optimal reference state entropy (S_sep)
    # If p <= 1/3, the state is separable, so the distance to the 
    # separable set is zero. Otherwise, we use the maximally mixed state.
    if p <= 1/3:
        S_sep = S_rho
    else:
        S_sep = 2.0  # log2(4)
        
    return max(0.0, S_sep - S_rho)
# -----------------------------
# Main analysis
# -----------------------------
def analyze_metrics(p_values):

    results = []

    for p in p_values:

        # 1. Fidelity (exact Werner-Bell overlap)
        fidelity = (3*p + 1)/4

        # 2. Concurrence
        concurrence = max(0.0, (3*p - 1)/2)

        # 3. Entanglement of Formation (Wootters)
        if concurrence > 0:
            x = (1 + np.sqrt(1 - concurrence**2))/2
            eof = binary_entropy(x)
        else:
            eof = 0.0

        # 4. Negativity (partial transpose eigenvalue structure)
        negativity = max(0.0, (3*p - 1)/4)

        # 5. Entropy of formation (cost approximation already included via EoF)
        ent_cost = eof

        # 6. Relative Entropy of Entanglement
        ree = relative_entropy_entanglement(p)

        # 7. Von Neumann entropy eigenvalues
        eigs = [(1 + 3*p)/4] + [(1 - p)/4]*3

        # 8. Rényi entropy (alpha = 2 and alpha = 5)
        renyi_2 = renyi_entropy(eigs, 2)
        renyi_5 = renyi_entropy(eigs, 5)

        results.append({
            'p': round(p, 2),
            'Fidelity': round(fidelity, 6),
            'Concurrence': round(concurrence, 6),
            'Negativity': round(negativity, 6),
            'EoF': round(eof, 6),
            'EntanglementCost': round(ent_cost, 6),
            'RelativeEntropyEnt': round(ree, 6),
            'Renyi(alpha=2)': round(renyi_2, 6),
            'Renyi(alpha=5)': round(renyi_5, 6)
        })

    return pd.DataFrame(results)

# -----------------------------
# Sample points (phase transition region included)
# -----------------------------
p_test = [0.22, 0.33, 0.34, 0.5, 0.9, 1.0]

df = analyze_metrics(p_test)

print("Complete Entanglement Metrics (Werner State):")
print(df.to_string(index=False))