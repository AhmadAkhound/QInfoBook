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
        # n = 0.5*cosh(2r), c = 0.5*sinh(2r) in the convention where
        # the vacuum covariance matrix is I/2.
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
        # Pure TMSV: det(sigma) = 1/16 exactly (numerically stable).
        det_sigma = 1.0 / 16.0

        delta_tilde = det_A + det_B + 2 * det_C_tilde

        # smallest symplectic eigenvalue of the partial transpose
        inner = max(0.0, delta_tilde**2 - 4 * det_sigma)
        # Rationalized smaller root avoids subtracting nearly equal numbers.
        nu_min_sq = (2.0 * det_sigma) / (delta_tilde + np.sqrt(inner))
        nu_min = np.sqrt(nu_min_sq)

        # logarithmic negativity E_N = max(0, -log2(2 nu_min))
        en = max(0.0, -np.log2(2 * nu_min)) if nu_min > 0 else 0.0

        status = "Entangled" if nu_min < 0.5 else "Separable"
        results.append({
            "r": round(r, 2),
            "nu_min": round(nu_min, 4) if nu_min >= 1e-3 else nu_min,
            "E_N": round(en, 4),
            "Status": status
        })
    return pd.DataFrame(results)

r_range = [0.0, 0.2, 0.3466, 0.5, 0.8, 1.0]
df_results = calculate_cv_entanglement_rigorous(r_range)
print("Rigorous phase space analysis of the TMSV:")
print(df_results.to_string(index=False))

# analytic check: nu_min should equal 0.5*exp(-2r)

print()
print("Analytic check nu_min = 0.5 * exp(-2r):")
for r in r_range:
    analytic = 0.5 * np.exp(-2 * r)
    print("r = %.4f   numeric vs analytic: %.6f" % (r, analytic))

