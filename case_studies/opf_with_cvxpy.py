import torch

# ============================================================
# 0. Utilities: evaluate objective / constraints (for reporting)
# ============================================================

def evaluate_all(v, data):

    G = data["G"]
    Gv = G @ v
    p = v * Gv
    f = torch.dot(v, Gv)

    # voltage
    V_lower = data["V_lower"]
    V_upper = data["V_upper"]
    v_viol_low = torch.clamp(V_lower - v, min=0.0)
    v_viol_up = torch.clamp(v - V_upper, min=0.0)
    max_v_viol = torch.max(torch.cat([v_viol_low, v_viol_up], dim=0)).item()

    # load / gen
    load_idx = data["load_idx"]
    gen_idx = data["gen_idx"]
    d = data["d"]
    p_cap = data["p_cap"]

    load_viol = torch.clamp(d[load_idx] - p[load_idx], min=0.0)
    gen_viol = torch.clamp(p[gen_idx] - p_cap[gen_idx], min=0.0)
    max_load_viol = load_viol.max().item() if load_viol.numel() else 0.0
    max_gen_viol = gen_viol.max().item() if gen_viol.numel() else 0.0

    # line
    edges_i = data["edges_i"]
    edges_j = data["edges_j"]
    g_ij = data["g_ij"]
    I_max = data["I_max"]
    current = g_ij * (v[edges_i] - v[edges_j])
    abs_current = torch.abs(current)
    line_viol = torch.clamp(abs_current - I_max, min=0.0)
    max_line_viol = line_viol.max().item() if line_viol.numel() else 0.0

    return {
        "f": f.item(),
        "p": p.detach().clone(),
        "abs_current": abs_current.detach().clone(),
        "max_v_viol": max_v_viol,
        "max_load_viol": max_load_viol,
        "max_gen_viol": max_gen_viol,
        "max_line_viol": max_line_viol,
    }


# ==============================
# 1. Generate test case
# ==============================

def build_test_case(
    N=5,
    V_min=0.5,
    V_max=1.5,
    seed=0,
):
    torch.manual_seed(seed)

    nodes = torch.arange(N)

    # chain network edges E = {(0,1), (1,2), ..., (N-2,N-1)}
    edges_i = []
    edges_j = []
    for i in range(N - 1):
        edges_i.append(i)
        edges_j.append(i + 1)
    edges_i = torch.tensor(edges_i, dtype=torch.long)
    edges_j = torch.tensor(edges_j, dtype=torch.long)
    E = len(edges_i)

    # conductances g_ij > 0
    g_ij = 0.5 + torch.rand(E)  # in (0.5, 1.5)

    # Laplacian G
    G = torch.zeros(N, N)
    for k in range(E):
        i = edges_i[k].item()
        j = edges_j[k].item()
        g = g_ij[k].item()
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g

    # voltage bounds
    V_lower = torch.full((N,), V_min)
    V_upper = torch.full((N,), V_max)

    # load / gen split
    num_load = N // 2
    load_idx = torch.arange(num_load, dtype=torch.long)
    gen_idx = torch.arange(num_load, N, dtype=torch.long)

    # load demand d_i > 0 for load buses
    d = torch.zeros(N)
    d[load_idx] = 0.01 + 0.04 * torch.rand(len(load_idx))

    # generation capacity p_cap > 0 for gen buses
    p_cap = torch.zeros(N)
    p_cap[gen_idx] = 1.0 + torch.rand(len(gen_idx))  # in (1,2)

    # line current limit I_max
    I_max = 1.0 + 0.5 * torch.rand(E)  # in (1,1.5)

    data = {
        "N": N,
        "nodes": nodes,
        "edges_i": edges_i,
        "edges_j": edges_j,
        "E": E,
        "g_ij": g_ij,
        "G": G,
        "V_lower": V_lower,
        "V_upper": V_upper,
        "load_idx": load_idx,
        "gen_idx": gen_idx,
        "d": d,
        "p_cap": p_cap,
        "I_max": I_max,
    }
    return data

import torch

def build_test_case_guaranteed_feasible(
    N=5,
    V_min=0.5,
    V_max=1.5,
    seed=0,
    load_positive=True,
    load_margin=0.8,   # d_i = load_margin * p_i(v*)
    gen_margin=0.2,    # p_cap_i = p_i(v*) + gen_margin
):
    '''

    Build a feasiable sample
    '''
    torch.manual_seed(seed)

    nodes = torch.arange(N)

    # chain edges
    edges_i = torch.arange(N-1, dtype=torch.long)
    edges_j = torch.arange(1, N, dtype=torch.long)
    E = N - 1

    # conductances
    g_ij = 0.5 + torch.rand(E)  # (0.5, 1.5)

    # Laplacian G
    G = torch.zeros(N, N)
    for k in range(E):
        i = edges_i[k].item()
        j = edges_j[k].item()
        g = g_ij[k].item()
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g

    # voltage bounds
    V_lower = torch.full((N,), V_min)
    V_upper = torch.full((N,), V_max)

    # load/gen split
    num_load = N // 2
    load_idx = torch.arange(num_load, dtype=torch.long)
    gen_idx = torch.arange(num_load, N, dtype=torch.long)


    I_max = 1.0 + 0.5 * torch.rand(E)  # (1, 1.5)


    dv_max = I_max / g_ij


    v_star = torch.empty(N)
    v_star[0] = 0.5 * (V_min + V_max)


    sign = 1.0
    for k in range(E):
        step = 0.5 * dv_max[k].item()
        v_star[k+1] = v_star[k] + sign * step

        if v_star[k+1] > V_max:
            sign = -1.0
            v_star[k+1] = min(V_max, v_star[k] + sign * step)
        if v_star[k+1] < V_min:
            sign = 1.0
            v_star[k+1] = max(V_min, v_star[k] + sign * step)


    v_star = torch.clamp(v_star, V_min, V_max)

    p_star = v_star * (G @ v_star)


    d = torch.zeros(N)
    p_cap = torch.zeros(N)


    if load_positive:
        v_star2 = v_star.clone()
        v_star2[load_idx] = V_max
        for k in range(E):
            i = edges_i[k].item()
            j = edges_j[k].item()
            diff = v_star2[i] - v_star2[j]
            bound = dv_max[k]
            if diff > bound:
                v_star2[i] = v_star2[j] + bound
            if diff < -bound:
                v_star2[i] = v_star2[j] - bound
        v_star2 = torch.clamp(v_star2, V_min, V_max)
        p_star2 = v_star2 * (G @ v_star2)

        v_star = v_star2
        p_star = p_star2


        d[load_idx] = load_margin * torch.clamp(p_star[load_idx], min=1e-6)
    else:
        d[load_idx] = load_margin * torch.clamp(p_star[load_idx], min=0.0)


    p_cap[gen_idx] = p_star[gen_idx] + gen_margin
    p_cap[gen_idx] = torch.clamp(p_cap[gen_idx], min=1e-6)


    I_max = torch.maximum(I_max, torch.abs(g_ij * (v_star[edges_i] - v_star[edges_j])) + 1e-6)

    data = {
        "N": N,
        "nodes": nodes,
        "edges_i": edges_i,
        "edges_j": edges_j,
        "E": E,
        "g_ij": g_ij,
        "G": G,
        "V_lower": V_lower,
        "V_upper": V_upper,
        "load_idx": load_idx,
        "gen_idx": gen_idx,
        "d": d,
        "p_cap": p_cap,
        "I_max": I_max
    }
    return data

# ==============================
# 2. Lagrangian  L(v, λ, γ, μ)
# ==============================

def lagrangian(v, lam, gam, mu, data):
    """
    L(v, λ, γ, μ) = v^T G v
                  + Σ_{i∈L} λ_i (d_i - p_i(v))
                  + Σ_{i∈G} γ_i (p_i(v) - p̄_i)
                  + Σ_{(i,j)∈E} μ_ij ( |g_ij (v_i - v_j)| - I_ij^max )
    p_i(v) = v_i (Gv)_i
    """
    G = data["G"]
    d = data["d"]
    p_cap = data["p_cap"]
    load_idx = data["load_idx"]
    gen_idx = data["gen_idx"]
    edges_i = data["edges_i"]
    edges_j = data["edges_j"]
    g_ij = data["g_ij"]
    I_max = data["I_max"]

    Gv = torch.matmul(G, v)
    p = v * Gv

    f = torch.dot(v, Gv)

    lag_load = torch.dot(lam[load_idx], (d[load_idx] - p[load_idx]))
    lag_gen = torch.dot(gam[gen_idx], (p[gen_idx] - p_cap[gen_idx]))

    v_i = v[edges_i]
    v_j = v[edges_j]
    current_ij = g_ij * (v_i - v_j)
    abs_current_ij = torch.abs(current_ij)
    lag_line = torch.dot(mu, (abs_current_ij - I_max))

    L = f + lag_load + lag_gen + lag_line
    return L, f, p, abs_current_ij


# ==============================
# 3. Projections
# ==============================

def project_voltage(v, V_lower, V_upper):
    return torch.clamp(v, V_lower, V_upper)

def project_nonnegative(x):
    return torch.clamp(x, min=0.0)


# ==============================
# 4. Primal–Dual projected gradient
# ==============================

# def primal_dual_solve(
#     data,
#     num_iters=500,
#     tau=0.1,
#     eta=0.1,
#     verbose=True,
# ):
#     N = data["N"]
#     V_lower = data["V_lower"]
#     V_upper = data["V_upper"]
#     load_idx = data["load_idx"]
#     gen_idx = data["gen_idx"]
#
#     # init
#     v = (V_lower + V_upper) / 2.0
#     v = v.clone().detach().requires_grad_(True)
#
#     lam = torch.zeros(N)
#     gam = torch.zeros(N)
#     mu = torch.zeros(data["E"])
#
#     history = {"f": [], "lambda": [], "gamma": [], "mu": []}
#
#     for k in range(num_iters):
#         # primal step
#         if v.grad is not None:
#             v.grad.zero_()
#
#         L, f, p, abs_current = lagrangian(v, lam, gam, mu, data)
#         L.backward()
#
#         with torch.no_grad():
#             v_new = v - tau * v.grad
#             v_new = project_voltage(v_new, V_lower, V_upper)
#
#         v = v_new.clone().detach().requires_grad_(True)
#
#         # dual step uses v^{k+1}
#         with torch.no_grad():
#             Gv = data["G"] @ v
#             p_new = v * Gv
#
#             lam_new = lam.clone()
#             lam_new[load_idx] = project_nonnegative(
#                 lam[load_idx] + eta * (data["d"][load_idx] - p_new[load_idx])
#             )
#
#             gam_new = gam.clone()
#             gam_new[gen_idx] = project_nonnegative(
#                 gam[gen_idx] + eta * (p_new[gen_idx] - data["p_cap"][gen_idx])
#             )
#
#             v_i = v[data["edges_i"]]
#             v_j = v[data["edges_j"]]
#             current_ij = data["g_ij"] * (v_i - v_j)
#             abs_current_ij = torch.abs(current_ij)
#
#             mu_new = project_nonnegative(
#                 mu + eta * (abs_current_ij - data["I_max"])
#             )
#
#             lam, gam, mu = lam_new, gam_new, mu_new
#
#         history["f"].append(f.item())
#         history["lambda"].append(lam.clone())
#         history["gamma"].append(gam.clone())
#         history["mu"].append(mu.clone())
#
#         if verbose and (k % 50 == 0 or k == num_iters - 1):
#             stats = evaluate_all(v.detach(), data)
#             print(
#                 f"Iter {k:4d}: f={stats['f']:.6f}, "
#                 f"V_viol={stats['max_v_viol']:.2e}, "
#                 f"load_viol={stats['max_load_viol']:.2e}, "
#                 f"gen_viol={stats['max_gen_viol']:.2e}, "
#                 f"line_viol={stats['max_line_viol']:.2e}"
#             )
#
#     return v.detach(), lam, gam, mu, history

# ============================== #
# 4. Primal–Dual projected gradient (PURE PyTorch, no autograd)
# ============================== #

def primal_dual_solve(
    data,
    num_iters=500,
    tau=0.1,
    eta=0.1,
    verbose=True,
):
    """
    Same interface/behavior as before, but:
    - primal step uses a manual gradient of L wrt v (no autograd).
    - dual steps remain the same.
    """
    N = data["N"]
    V_lower = data["V_lower"]
    V_upper = data["V_upper"]
    load_idx = data["load_idx"]
    gen_idx = data["gen_idx"]

    edges_i = data["edges_i"]
    edges_j = data["edges_j"]
    g_ij = data["g_ij"]
    I_max = data["I_max"]
    G = data["G"]

    # init
    v = (V_lower + V_upper) / 2.0
    lam = torch.zeros(N, dtype=v.dtype, device=v.device)
    gam = torch.zeros(N, dtype=v.dtype, device=v.device)
    mu  = torch.zeros(data["E"], dtype=v.dtype, device=v.device)

    history = {"f": [], "lambda": [], "gamma": [], "mu": []}

    # helper: manual gradient wrt v
    def grad_L(v, lam, gam, mu):
        # f = v^T G v  -> grad = 2 G v  (G is symmetric Laplacian)
        Gv = G @ v
        grad_f = 2.0 * Gv

        # p_i(v) = v_i (Gv)_i
        # load term: sum_{i in L} lam_i (d_i - p_i) contributes -lam_i * p_i
        # gen  term: sum_{i in G} gam_i (p_i - pcap_i) contributes +gam_i * p_i
        # combine as a_i:
        a = torch.zeros_like(v)
        a[load_idx] = -lam[load_idx]
        a[gen_idx]  =  gam[gen_idx]

        # q(v) = sum_i a_i v_i (Gv)_i = v^T diag(a) G v
        # grad q = (diag(a) G + G diag(a)) v
        grad_p_part = (a * (G @ v)) + (G @ (a * v))

        # line term: sum_k mu_k * | g_k (v_i - v_j) |
        # subgradient wrt v_i: + mu_k * g_k * sign(v_i - v_j)
        #              v_j: - mu_k * g_k * sign(v_i - v_j)
        vi = v[edges_i]
        vj = v[edges_j]
        diff = vi - vj

        # sign(0) -> 0 (one valid subgradient choice)
        sgn = torch.sign(diff)

        w = mu * g_ij * sgn  # shape [E]
        grad_line = torch.zeros_like(v)
        grad_line.scatter_add_(0, edges_i, w)
        grad_line.scatter_add_(0, edges_j, -w)

        return grad_f + grad_p_part + grad_line

    for k in range(num_iters):
        # record objective f at current v (like before)
        f = torch.dot(v, G @ v)

        # primal step (manual grad)
        with torch.no_grad():
            g = grad_L(v, lam, gam, mu)
            v_new = v - tau * g
            v_new = project_voltage(v_new, V_lower, V_upper)
            v = v_new

        # dual step uses v^{k+1} (same as before)
        with torch.no_grad():
            Gv = G @ v
            p_new = v * Gv

            lam_new = lam.clone()
            lam_new[load_idx] = project_nonnegative(
                lam[load_idx] + eta * (data["d"][load_idx] - p_new[load_idx])
            )

            gam_new = gam.clone()
            gam_new[gen_idx] = project_nonnegative(
                gam[gen_idx] + eta * (p_new[gen_idx] - data["p_cap"][gen_idx])
            )

            vi = v[edges_i]
            vj = v[edges_j]
            current_ij = g_ij * (vi - vj)
            abs_current_ij = torch.abs(current_ij)

            mu_new = project_nonnegative(
                mu + eta * (abs_current_ij - I_max)
            )

            lam, gam, mu = lam_new, gam_new, mu_new

        history["f"].append(float(f.item()))
        history["lambda"].append(lam.clone())
        history["gamma"].append(gam.clone())
        history["mu"].append(mu.clone())

        if verbose and (k % 100 == 0 or k == num_iters - 1):
            stats = evaluate_all(v.detach(), data)
            print(
                f"Iter {k:4d}: f={stats['f']:.6f}, "
                f"V_viol={stats['max_v_viol']:.2e}, "
                f"load_viol={stats['max_load_viol']:.2e}, "
                f"gen_viol={stats['max_gen_viol']:.2e}, "
                f"line_viol={stats['max_line_viol']:.2e}"
            )

    return v.detach(), lam, gam, mu, history


# ============================================================
# 5. CVXPY baselines (DCP-safe): objective via incidence matrix
# ============================================================

def _build_weighted_incidence(data):
    """
    Build weighted incidence A in R^{E x N} such that:
        sum_{(i,j)} g_ij (v_i - v_j)^2 = || A v ||_2^2
    where each row k for edge (i,j):
        A[k,i] = +sqrt(g_ij),  A[k,j] = -sqrt(g_ij)
    """
    import numpy as np

    N = int(data["N"])
    E = int(data["E"])
    edges_i = data["edges_i"].detach().cpu().numpy().astype(int)
    edges_j = data["edges_j"].detach().cpu().numpy().astype(int)
    g_ij = data["g_ij"].detach().cpu().numpy()

    A = np.zeros((E, N))
    w = np.sqrt(g_ij)
    for k in range(E):
        i = edges_i[k]
        j = edges_j[k]
        A[k, i] = +w[k]
        A[k, j] = -w[k]
    return A


def cvxpy_baseline_qp(data, solver="OSQP", verbose=False):
    import cvxpy as cp
    import numpy as np

    N = int(data["N"])
    V_lower = data["V_lower"].detach().cpu().numpy()
    V_upper = data["V_upper"].detach().cpu().numpy()

    edges_i = data["edges_i"].detach().cpu().numpy().astype(int)
    edges_j = data["edges_j"].detach().cpu().numpy().astype(int)
    g_ij = data["g_ij"].detach().cpu().numpy()
    I_max = data["I_max"].detach().cpu().numpy()
    E = int(data["E"])

    A = _build_weighted_incidence(data)

    v = cp.Variable(N)

    # DCP-safe convex objective:
    obj = cp.Minimize(cp.sum_squares(A @ v))

    cons = [v >= V_lower, v <= V_upper]
    for k in range(E):
        i = int(edges_i[k]); j = int(edges_j[k])
        cons.append(cp.abs(g_ij[k] * (v[i] - v[j])) <= I_max[k])

    prob = cp.Problem(obj, cons)

    if solver.upper() == "OSQP":
        prob.solve(solver=cp.OSQP, verbose=verbose)
    elif solver.upper() == "SCS":
        prob.solve(solver=cp.SCS, verbose=verbose)
    elif solver.upper() == "ECOS":
        prob.solve(solver=cp.ECOS, verbose=verbose)
    else:
        prob.solve(verbose=verbose)

    if v.value is None:
        raise RuntimeError(f"CVXPY-QP failed: status={prob.status}")

    v_star = torch.tensor(v.value, dtype=data["V_lower"].dtype)
    return v_star, float(prob.value), prob.status


def _build_Pi_matrices(G_torch):
    import numpy as np
    G = G_torch.detach().cpu().numpy()
    N = G.shape[0]
    P = []
    for i in range(N):
        Pi = np.zeros((N, N))
        Pi[i, i] = G[i, i]
        for j in range(N):
            if j != i and G[i, j] != 0.0:
                Pi[i, j] = 0.5 * G[i, j]
                Pi[j, i] = 0.5 * G[i, j]
        P.append(Pi)
    return P


def cvxpy_baseline_sca(
    data,
    v0=None,
    num_sca_iters=40,
    rho=1e-2,
    solver="OSQP",
    verbose=False,
):
    import cvxpy as cp
    import numpy as np

    N = int(data["N"])
    V_lower = data["V_lower"].detach().cpu().numpy()
    V_upper = data["V_upper"].detach().cpu().numpy()

    load_idx = data["load_idx"].detach().cpu().numpy().astype(int)
    gen_idx = data["gen_idx"].detach().cpu().numpy().astype(int)
    d = data["d"].detach().cpu().numpy()
    p_cap = data["p_cap"].detach().cpu().numpy()

    edges_i = data["edges_i"].detach().cpu().numpy().astype(int)
    edges_j = data["edges_j"].detach().cpu().numpy().astype(int)
    g_ij = data["g_ij"].detach().cpu().numpy()
    I_max = data["I_max"].detach().cpu().numpy()
    E = int(data["E"])

    A = _build_weighted_incidence(data)
    P_list = _build_Pi_matrices(data["G"])

    if v0 is None:
        v_prev = ((V_lower + V_upper) / 2.0).copy()
    else:
        v_prev = v0.detach().cpu().numpy().copy()

    last_prob = None

    for t in range(num_sca_iters):
        v = cp.Variable(N)

        # DCP-safe convex objective: ||A v||^2 + proximal term
        obj = cp.sum_squares(A @ v) + 0.5 * rho * cp.sum_squares(v - v_prev)
        cons = [v >= V_lower, v <= V_upper]

        # line limits
        for k in range(E):
            i = int(edges_i[k]); j = int(edges_j[k])
            cons.append(cp.abs(g_ij[k] * (v[i] - v[j])) <= I_max[k])

        # linearized load constraints: p_lin >= d
        for i in load_idx:
            Pi = P_list[i]
            p0 = float(v_prev.T @ Pi @ v_prev)
            grad = 2.0 * (Pi @ v_prev)
            p_lin = p0 + grad @ (v - v_prev)
            cons.append(p_lin >= d[i])

        # linearized gen constraints: p_lin <= p_cap
        for i in gen_idx:
            Pi = P_list[i]
            p0 = float(v_prev.T @ Pi @ v_prev)
            grad = 2.0 * (Pi @ v_prev)
            p_lin = p0 + grad @ (v - v_prev)
            cons.append(p_lin <= p_cap[i])

        prob = cp.Problem(cp.Minimize(obj), cons)

        if solver.upper() == "OSQP":
            prob.solve(solver=cp.OSQP, verbose=verbose)
        elif solver.upper() == "SCS":
            prob.solve(solver=cp.SCS, verbose=verbose)
        elif solver.upper() == "ECOS":
            prob.solve(solver=cp.ECOS, verbose=verbose)
        else:
            prob.solve(verbose=verbose)

        if v.value is None:
            raise RuntimeError(f"CVXPY-SCA failed at iter {t}: status={prob.status}")

        v_prev = v.value
        last_prob = prob

    v_star = torch.tensor(v_prev, dtype=data["V_lower"].dtype)
    return v_star, float(last_prob.value), last_prob.status


# ==============================
# 6. Main: run + compare
# ==============================

if __name__ == "__main__":
    data = build_test_case(N=5, V_min=0.9, V_max=1.2, seed=0)
    #data = build_test_case_guaranteed_feasible(N=20, V_min=0.6, V_max=1.5, seed=0)

    print("=== Primal–Dual (PyTorch autograd) ===")
    v_pd, lam_opt, gam_opt, mu_opt, history = primal_dual_solve(
        data, num_iters=8000, tau=0.06, eta=0.06, verbose=True
    )
    stats_pd = evaluate_all(v_pd, data)
    print(f"[PD] f={stats_pd['f']:.6f}, "
          f"V_viol={stats_pd['max_v_viol']:.2e}, "
          f"load_viol={stats_pd['max_load_viol']:.2e}, "
          f"gen_viol={stats_pd['max_gen_viol']:.2e}, "
          f"line_viol={stats_pd['max_line_viol']:.2e}")


    print("\n=== CVXPY Baseline 1: SCA (linearize p_i constraints) ===")
    try:
        v_sca, obj_sca, status_sca = cvxpy_baseline_sca(
            data,
            v0=v_pd,
            num_sca_iters=30,
            rho=1e-2,
            solver="OSQP",
            verbose=False,
        )
        stats_sca = evaluate_all(v_sca, data)
        print("[CVX-SCA] status:", status_sca)
        print(f"[CVX-SCA] f={stats_sca['f']:.6f}, "
              f"V_viol={stats_sca['max_v_viol']:.2e}, "
              f"load_viol={stats_sca['max_load_viol']:.2e}, "
              f"gen_viol={stats_sca['max_gen_viol']:.2e}, "
              f"line_viol={stats_sca['max_line_viol']:.2e}")
        print(f"||v_cvx - v_pd||={torch.linalg.norm(v_sca-v_pd):6e}")
    except Exception as e:
        print("CVXPY-SCA failed:", e)

