"""
========================================================================
Title       : Entanglement of the Two-Mode Squeezed Vacuum in Phase Space
Book        : Quantum Entanglement, Volume 1
Chapter     : 7  --  Continuous Variables and Correlations Beyond Entanglement
Code Ref    : phase_analysis
Description : Builds the covariance matrix of the two-mode squeezed vacuum and
              detects its entanglement through the Simon criterion and the
              logarithmic negativity from the partial-transpose symplectic
              eigenvalue.
========================================================================
"""
import numpy as np
import pandas as pd

def calculate_cv_entanglement_rigorous(r_values):
    # Rigorous CV entanglement for the two-mode squeezed vacuum state,
    # based on the symplectic invariants of the covariance matrix.
    results = []
    for r in r_values:
        # covariance matrix elements of the TMSV
        # n = cosh(2r), c = sinh(2r) in the convention where vacuum = I
        n = 0.5 * np.cosh(2 * r)
        c = 0.5 * np.sinh(2 * r)

        # local blocks A, B and correlation block C
        A = np.array([[n, 0.0], [0.0, n]])
        B = np.array([[n, 0.0], [0.0, n]])
        C = np.array([[c, 0.0], [0.0, -c]])

        # full 4x4 covariance matrix
        sigma = np.block([[A, C], [C, B]])

        # partial transpose on mode 2 (p2 -> -p2) turns C into diag(c, c)
        C_tilde = np.array([[c, 0.0], [0.0, c]])
        sigma_pt = np.block([[A, C_tilde], [C_tilde, B]])

        # symplectic invariant of the partial transpose
        det_A = np.linalg.det(A)
        det_B = np.linalg.det(B)
        det_C_tilde = np.linalg.det(C_tilde)
        det_sigma = np.linalg.det(sigma)   # invariant under partial transpose

        delta_tilde = det_A + det_B + 2 * det_C_tilde

        # smallest symplectic eigenvalue of the partial transpose
        inner = max(0.0, delta_tilde**2 - 4 * det_sigma)
        nu_min_sq = 0.5 * (delta_tilde - np.sqrt(inner))
        nu_min = np.sqrt(max(0.0, nu_min_sq))

        # logarithmic negativity E_N = max(0, -log2(2 nu_min))
        en = max(0.0, -np.log2(2 * nu_min)) if nu_min > 0 else 0.0

        status = "Entangled" if nu_min < 0.5 else "Separable"
        results.append({
            "r": round(r, 2),
            "nu_min": round(nu_min, 4),
            "E_N": round(en, 4),
            "Status": status
        })
    return pd.DataFrame(results)

r_range = [0.0, 0.2, 0.3466, 0.5, 0.8, 1.0]
df_results = calculate_cv_entanglement_rigorous(r_range)
print("Rigorous phase space analysis of the TMSV:")
print(df_results.to_string(index=False))

# analytic check: nu_min should equal exp(-2r)
print()
print("Analytic check nu_min = exp(-2r):")
for r in r_range:
    print("r = %.4f   numeric vs analytic: %.6f" % (r, np.exp(-2 * r)))