"""
========================================================================
Title       : Squashed Entanglement via Channel Extensions
Book        : Quantum Entanglement, Volume 1
Chapter     : 4  --  Operational Measures and the Cost of Entanglement
Code Ref    : squashed_extension
Description : Computes upper bounds on the squashed entanglement of the Werner
              state from the trivial and the flag extensions, comparing their
              tightness via the conditional mutual information.
========================================================================
"""
import numpy as np


def vn_entropy(rho):
    """von Neumann entropy in bits."""
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return -np.sum(ev * np.log2(ev))


def ptrace(rho, keep, dims):
    """Partial trace keeping subsystems in `keep`."""
    n = len(dims)
    rho = rho.reshape(dims + dims)
    trace_out = [i for i in range(n) if i not in keep]
    for ax in sorted(trace_out, reverse=True):
        rho = np.trace(rho, axis1=ax, axis2=ax + rho.ndim // 2)
    d = int(np.prod([dims[i] for i in keep]))
    return rho.reshape(d, d)


# Werner state base: Bell state |Phi+> and white noise
phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
Bell = np.outer(phi, phi)
mix = np.eye(4) / 4

print(f"{'p':<6} | {'upper (trivial ext.)':<22} | {'upper (flag ext.)':<18}")
print("-" * 52)

for p in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    rho = p * Bell + (1 - p) * mix

    # Trivial extension: E is uncorrelated, CMI reduces to I(A:B)/2
    rA = ptrace(rho, [0], [2, 2])
    rB = ptrace(rho, [1], [2, 2])
    upper_trivial = 0.5 * (vn_entropy(rA) + vn_entropy(rB) - vn_entropy(rho))

    # Flag extension: E records which component (Bell vs. noise)
    e0 = np.array([1, 0])
    e1 = np.array([0, 1])
    rho_ABE = (p * np.kron(Bell, np.outer(e0, e0))
               + (1 - p) * np.kron(mix, np.outer(e1, e1)))
    rAE = ptrace(rho_ABE, [0, 2], [2, 2, 2])
    rBE = ptrace(rho_ABE, [1, 2], [2, 2, 2])
    rE = ptrace(rho_ABE, [2], [2, 2, 2])
    cmi = vn_entropy(rAE) + vn_entropy(rBE) - vn_entropy(rho_ABE) - vn_entropy(rE)
    upper_flag = 0.5 * cmi

    print(f"{p:<6.2f} | {upper_trivial:<22.4f} | {upper_flag:<18.4f}")

print("\nNote: the trivial extension gives a tighter (smaller) upper bound here,")
print("showing that a naive flag extension fails to squash classical correlations.")