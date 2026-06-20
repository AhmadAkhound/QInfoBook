import numpy as np

# ============================================================
# Von Neumann entropy
# ============================================================

def entropy(rho):
    eig = np.linalg.eigvalsh(rho)
    eig = np.real_if_close(eig)
    eig = eig[eig > 1e-12]
    if len(eig) == 0:
        return 0.0
    return float(-np.sum(eig * np.log2(eig)))

# ============================================================
# Partial traces (qubit A is the first, B is the second)
# ============================================================

def ptrace_A(rho):
    rhoB = np.zeros((2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            for bp in range(2):
                i = 2*a + b
                j = 2*a + bp
                rhoB[b, bp] += rho[i, j]
    return rhoB

def ptrace_B(rho):
    rhoA = np.zeros((2, 2), dtype=complex)
    for a in range(2):
        for ap in range(2):
            for b in range(2):
                i = 2*a + b
                j = 2*ap + b
                rhoA[a, ap] += rho[i, j]
    return rhoA

# ============================================================
# Conditional states of B after a measurement on A
# ============================================================

def conditional_states(rho, projectors):
    I = np.eye(2)
    probs, states = [], []
    for P in projectors:
        M = np.kron(P, I)
        rho_post = M @ rho @ M
        p = np.trace(rho_post).real
        probs.append(p)
        if p < 1e-12:
            states.append(np.zeros((2, 2)))
        else:
            rho_post /= p
            states.append(ptrace_A(rho_post))
    return probs, states

# ============================================================
# Holevo information I(Q;B) for a measurement basis on A
# ============================================================

def holevo_information(rho, projectors):
    rhoB = ptrace_A(rho)
    HB = entropy(rhoB)
    probs, states = conditional_states(rho, projectors)
    average = sum(p * entropy(s) for p, s in zip(probs, states))
    return HB - average, average

# ============================================================
# Conditional entropy H(Q|B) after measurement on A in a basis
# ============================================================

def conditional_measurement_entropy(rho, projectors):
    rhoB = ptrace_A(rho)
    HB = entropy(rhoB)
    probs, states = conditional_states(rho, projectors)
    Hp = sum(-p * np.log2(p) for p in probs if p > 1e-12)
    average = sum(p * entropy(s) for p, s in zip(probs, states))
    return Hp + average - HB

# ============================================================
# Measurement bases: Pauli Z and Pauli X
# ============================================================

proj0 = np.array([[1, 0], [0, 0]], dtype=complex)
proj1 = np.array([[0, 0], [0, 1]], dtype=complex)
plus = np.array([1, 1]) / np.sqrt(2)
minus = np.array([1, -1]) / np.sqrt(2)
proj_plus = np.outer(plus, plus)
proj_minus = np.outer(minus, minus)
Z = [proj0, proj1]
X = [proj_plus, proj_minus]
c = 0.5  # maximal overlap for Z and X bases

# ============================================================
# Full analysis: Berta bound vs corrected Adabi bound
# Adabi delta = I(A;B) - [I(Q;B) + I(R;B)]   (correct definition)
# ============================================================

def analyze_state(rho, label):
    print(f"\n========== {label} ==========")

    rhoA = ptrace_B(rho)
    rhoB = ptrace_A(rho)

    SA = entropy(rhoA)
    SB = entropy(rhoB)
    SAB = entropy(rho)

    S_AgB = SAB - SB
    I_AB = SA + SB - SAB

    IX, _ = holevo_information(rho, X)
    IZ, _ = holevo_information(rho, Z)

    HXB = conditional_measurement_entropy(rho, X)
    HZB = conditional_measurement_entropy(rho, Z)
    actual_sum = HXB + HZB

    berta_bound = np.log2(1 / c) + S_AgB

    delta = I_AB - (IX + IZ)
    adabi_bound = berta_bound + max(0, delta)

    print(f"S(A|B)           = {S_AgB:.5f}")
    print(f"I(A;B)           = {I_AB:.5f}")
    print(f"I(Z;B), I(X;B)   = {IZ:.5f}, {IX:.5f}")
    print(f"delta            = {delta:.5f}")
    print(f"H(Z|B), H(X|B)   = {HZB:.5f}, {HXB:.5f}")
    print(f"Actual sum       = {actual_sum:.5f}")
    print(f"Berta bound      = {berta_bound:.5f}")
    print(f"Adabi bound      = {adabi_bound:.5f}")


# ------------------------------------------------------------
# Example 1: maximal Bell state (pure, symmetric)
# ------------------------------------------------------------
phi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(phi, phi.conj())
analyze_state(rho_bell, "Maximal Bell state |Phi+>")

# ------------------------------------------------------------
# Example 2: asymmetric pure state alpha|00>+beta|11>
# ------------------------------------------------------------
theta = np.pi / 8
alpha, beta = np.cos(theta), np.sin(theta)
psi = np.array([alpha, 0, 0, beta], dtype=complex)
rho_asym = np.outer(psi, psi.conj())
analyze_state(rho_asym, "Asymmetric pure state alpha|00>+beta|11>")

# ------------------------------------------------------------
# Example 3: noisy mixed state p|Phi+><Phi+| + (1-p)|01><01|
# ------------------------------------------------------------
p = 0.8
ket01 = np.array([0, 1, 0, 0], dtype=complex)
rho01 = np.outer(ket01, ket01.conj())
rho_mixed = p * rho_bell + (1 - p) * rho01
analyze_state(rho_mixed, "Noisy mixed state p=0.8")
