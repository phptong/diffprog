import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any


# ====================== Input set intersection ======================

def intersect_input_box_linf(
    x_nom: torch.Tensor,
    eps: float,
    l0_box: torch.Tensor,
    u0_box: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    lower = torch.max(l0_box, x_nom - eps)
    upper = torch.min(u0_box, x_nom + eps)
    return lower, upper

def project_box(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)

def project_input_set(
    x0: torch.Tensor,
    x_nom: torch.Tensor,
    eps: float,
    l0_box: torch.Tensor,
    u0_box: torch.Tensor
) -> torch.Tensor:
    """
    Π_{S_in ∩ [l0,u0]} implemented by coordinate-wise clipping
    """
    lower, upper = intersect_input_box_linf(x_nom, eps, l0_box, u0_box)
    return project_box(x0, lower, upper)


# ====================== IBP bounds ======================

def linear_interval(W: torch.Tensor, b: torch.Tensor,
                    l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if l.dim() == 1:
        l = l.unsqueeze(0)
        u = u.unsqueeze(0)

    W_pos = torch.clamp(W, min=0.0)
    W_neg = torch.clamp(W, max=0.0)

    y_l = l @ W_pos.t() + u @ W_neg.t() + b
    y_u = u @ W_pos.t() + l @ W_neg.t() + b
    return y_l, y_u

def relu_interval(l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.clamp(l, min=0.0), torch.clamp(u, min=0.0)

def compute_ibp_bounds(
    layers: List[nn.Module],
    x_nom: torch.Tensor,
    eps: float,
    l0_box: torch.Tensor,
    u0_box: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    SOUND bounds for each activation x_k (post-layer outputs),
    starting from x0 ∈ linf-ball ∩ [l0_box,u0_box].
    """
    l_bounds: List[torch.Tensor] = []
    u_bounds: List[torch.Tensor] = []

    l0, u0 = intersect_input_box_linf(x_nom, eps, l0_box, u0_box)
    l_bounds.append(l0)
    u_bounds.append(u0)

    l = l0
    u = u0
    for layer in layers:
        if isinstance(layer, nn.Linear):
            l, u = linear_interval(layer.weight, layer.bias, l, u)
        elif isinstance(layer, nn.ReLU):
            l, u = relu_interval(l, u)
        else:
            raise TypeError(f"Unsupported layer for IBP: {type(layer)}")
        l_bounds.append(l)
        u_bounds.append(u)

    return l_bounds, u_bounds


# ====================== Lagrangian (for monitoring only) ======================

@torch.no_grad()
def lagrangian_value(
    layers: List[nn.Module],
    xs: List[torch.Tensor],
    lambdas: List[torch.Tensor],
    c: torch.Tensor,
    d: float,
) -> float:
    """
    L(x,λ)=c^T xK + d + Σ λ_k^T (x_{k+1} - h_k(x_k))
    """
    K = len(layers)
    val = (c.flatten() * xs[K].flatten()).sum() + float(d)
    for k in range(K):
        hk = layers[k](xs[k])
        res = xs[k + 1] - hk
        val = val + (lambdas[k].flatten() * res.flatten()).sum()
    return float(val.item())


# ====================== Oracle g(lambda)=sup_x L(x,lambda) (optional evaluation) ======================

def _box_sup_linear(coeff: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_star = torch.where(coeff >= 0, upper, lower)
    val = (coeff.flatten() * x_star.flatten()).sum()
    return x_star, val


def _relu_sup_1d(a: float, b: float, l: float, u: float) -> Tuple[float, float, float]:
    """
    sup_{x in [l,u]} a*x - b*relu(x)
    returns (x_star, relu(x_star), sup_value)
    """
    if u <= 0.0:
        x_star = u if a > 0.0 else l
        return x_star, 0.0, a * x_star

    if l >= 0.0:
        coef = a - b
        x_star = u if coef > 0.0 else l
        return x_star, x_star, coef * x_star

    # crossing 0
    if a > 0.0:
        x1, v1 = 0.0, 0.0
    else:
        x1, v1 = l, a * l

    coef = a - b
    if coef > 0.0:
        x2, v2 = u, coef * u
    else:
        x2, v2 = 0.0, 0.0

    if v2 >= v1:
        return x2, max(0.0, x2), v2
    else:
        return x1, 0.0, v1


def _relu_sup_vector(a: torch.Tensor, b: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    a1 = a.flatten()
    b1 = b.flatten()
    l1 = lower.flatten()
    u1 = upper.flatten()

    a_cpu = a1.detach().cpu().tolist()
    b_cpu = b1.detach().cpu().tolist()
    l_cpu = l1.detach().cpu().tolist()
    u_cpu = u1.detach().cpu().tolist()

    vs = []
    for ai, bi, li, ui in zip(a_cpu, b_cpu, l_cpu, u_cpu):
        _, _, v_i = _relu_sup_1d(float(ai), float(bi), float(li), float(ui))
        vs.append(v_i)

    return torch.tensor(vs, dtype=a1.dtype, device=a1.device).sum()


@torch.no_grad()
def g_oracle(
    layers: List[nn.Module],
    x_nom: torch.Tensor,
    c: torch.Tensor,
    d: float,
    eps: float,
    l0_box: torch.Tensor,
    u0_box: torch.Tensor,
    l_bounds: List[torch.Tensor],
    u_bounds: List[torch.Tensor],
    lambdas: List[torch.Tensor],
    enforce_last_lambda_eq_minus_c: bool = True,
) -> float:
    """
    Evaluate g(λ)=sup_x L(x,λ) over x0 in S_in∩[l0,u0] and xk in [lk,uk].
    This is used ONLY for strict certificate checking (optional).
    """
    K = len(layers)
    lam = list(lambdas)
    if enforce_last_lambda_eq_minus_c:
        lam[K - 1] = -c.clone()

    # domain for x0
    lower0, upper0 = intersect_input_box_linf(x_nom, eps, l0_box, u0_box)

    g = torch.tensor(float(d), dtype=x_nom.dtype, device=x_nom.device)

    # xK term vanishes if enforce λ_{K-1}=-c (unbounded-xK style)
    # otherwise you should include sup over xK, but we stick to your paper constraint.

    # term for x0: sup -λ0^T h0(x0)
    lam0 = lam[0]
    h0 = layers[0]
    if isinstance(h0, nn.Linear):
        W, b = h0.weight, h0.bias
        coeff = -(W.t() @ lam0.flatten()).view_as(lower0)
        _, v = _box_sup_linear(coeff, lower0, upper0)
        g = g + v - (lam0.flatten() * b.flatten()).sum()
    elif isinstance(h0, nn.ReLU):
        a = torch.zeros_like(lower0)
        g = g + _relu_sup_vector(a, lam0, lower0, upper0)
    else:
        raise TypeError(f"Unsupported first layer type: {type(h0)}")

    # intermediate k=1..K-1: sup [λ_{k-1}^T x_k - λ_k^T h_k(x_k)]
    for k in range(1, K):
        hk = layers[k]
        lam_prev = lam[k - 1]
        lam_k = lam[k]
        lower, upper = l_bounds[k], u_bounds[k]

        if isinstance(hk, nn.Linear):
            W, b = hk.weight, hk.bias
            coeff = (lam_prev.flatten() - (W.t() @ lam_k.flatten())).view_as(lower)
            _, v = _box_sup_linear(coeff, lower, upper)
            g = g + v - (lam_k.flatten() * b.flatten()).sum()
        elif isinstance(hk, nn.ReLU):
            g = g + _relu_sup_vector(lam_prev, lam_k, lower, upper)
        else:
            raise TypeError(f"Unsupported layer type: {type(hk)}")

    return float(g.item())


# ====================== Paper-aligned Primal–Dual algorithm ======================

def dual_g_lamda(
    layers: List[nn.Module],
    x_nom: torch.Tensor,
    c: torch.Tensor,
    d: float,
    eps: float,
    l0_box: torch.Tensor,
    u0_box: torch.Tensor,
    l_bounds: List[torch.Tensor],
    u_bounds: List[torch.Tensor],
    tau_list: List[float],
    sigma_list: List[float],
    n_steps: int = 500,
    enforce_last_lambda_eq_minus_c: bool = True,
    strict_eval_oracle_every: int = 0,  # 0 disables oracle eval
) -> Dict[str, Any]:
    device = x_nom.device
    K = len(layers)
    assert len(l_bounds) == K + 1 and len(u_bounds) == K + 1
    assert len(tau_list) == K + 1 and len(sigma_list) == K

    # initialize primal x by forward from x_nom (feasible w.r.t equalities, then projected to bounds)
    xs: List[torch.Tensor] = [None] * (K + 1)
    xs[0] = project_input_set(x_nom.clone(), x_nom, eps, l0_box, u0_box)
    for k in range(K):
        xs[k + 1] = layers[k](xs[k])
        xs[k + 1] = project_box(xs[k + 1], l_bounds[k + 1], u_bounds[k + 1])

    # initialize dual lambdas (shape like x_{k+1})
    lambdas: List[torch.Tensor] = [None] * K
    for k in range(K - 1):
        lambdas[k] = torch.zeros_like(xs[k + 1], device=device)
    lambdas[K - 1] = (-c).clone() if enforce_last_lambda_eq_minus_c else torch.zeros_like(xs[K], device=device)

    best_g_oracle = float("inf")
    certified_by_oracle = False

    for t in range(n_steps):

        if strict_eval_oracle_every > 0 and ((t + 1) % strict_eval_oracle_every == 0):
            gval = g_oracle(
                layers=layers,
                x_nom=x_nom,
                c=c,
                d=d,
                eps=eps,
                l0_box=l0_box,
                u0_box=u0_box,
                l_bounds=l_bounds,
                u_bounds=u_bounds,
                lambdas=lambdas,
                enforce_last_lambda_eq_minus_c=True,
            )
            best_g_oracle = min(best_g_oracle, gval)
            if best_g_oracle < 0.0:
                certified_by_oracle = True

    return {
        "best_g_oracle": best_g_oracle,
        "certified_by_oracle": certified_by_oracle,
        "xs": xs,
        "lambdas": lambdas,
    }


# ====================== CVXPY baselines ======================

def cvxpy_triangle_lp_baseline(
    layers: List[nn.Module],
    c: torch.Tensor,
    d: float,
    l_bounds: List[torch.Tensor],
    u_bounds: List[torch.Tensor],
    solver: str = "ECOS",
    verbose: bool = False,
) -> Dict[str, Any]:
    import numpy as np
    import cvxpy as cp

    def to_np(t: torch.Tensor) -> "np.ndarray":
        return t.detach().cpu().numpy().astype("float64")

    K = len(layers)
    dims = [int(l_bounds[k].numel()) for k in range(K + 1)]
    x_vars = [cp.Variable(dims[k]) for k in range(K + 1)]
    cons = []

    for k in range(K + 1):
        lk = to_np(l_bounds[k]).reshape(-1)
        uk = to_np(u_bounds[k]).reshape(-1)
        cons += [x_vars[k] >= lk, x_vars[k] <= uk]

    for i, layer in enumerate(layers):
        x_in = x_vars[i]
        x_out = x_vars[i + 1]

        if isinstance(layer, nn.Linear):
            W = to_np(layer.weight)
            b = to_np(layer.bias).reshape(-1)
            cons += [x_out == W @ x_in + b]

        elif isinstance(layer, nn.ReLU):
            lz = to_np(l_bounds[i]).reshape(-1)
            uz = to_np(u_bounds[i]).reshape(-1)

            cons += [x_out >= 0, x_out >= x_in]
            for j in range(dims[i]):
                lj = float(lz[j]); uj = float(uz[j])
                if uj <= 0.0:
                    cons += [x_out[j] == 0.0]
                elif lj >= 0.0:
                    cons += [x_out[j] == x_in[j]]
                else:
                    slope = uj / (uj - lj)
                    cons += [x_out[j] <= slope * (x_in[j] - lj)]
        else:
            raise TypeError(f"Unsupported layer in LP baseline: {type(layer)}")

    c_np = to_np(c).reshape(-1)
    prob = cp.Problem(cp.Maximize(c_np @ x_vars[K] + float(d)), cons)
    val = prob.solve(solver=solver, verbose=verbose)

    return {"status": prob.status, "opt": float(val), "solver": solver, "num_cons": len(cons)}


def cvxpy_milp_exact_baseline(
    layers: List[nn.Module],
    c: torch.Tensor,
    d: float,
    l_bounds: List[torch.Tensor],
    u_bounds: List[torch.Tensor],
    solver: str = "ECOS_BB",
    verbose: bool = False,
) -> Dict[str, Any]:
    import numpy as np
    import cvxpy as cp

    def to_np(t: torch.Tensor) -> "np.ndarray":
        return t.detach().cpu().numpy().astype("float64")

    K = len(layers)
    dims = [int(l_bounds[k].numel()) for k in range(K + 1)]
    x_vars = [cp.Variable(dims[k]) for k in range(K + 1)]
    cons = []

    for k in range(K + 1):
        lk = to_np(l_bounds[k]).reshape(-1)
        uk = to_np(u_bounds[k]).reshape(-1)
        cons += [x_vars[k] >= lk, x_vars[k] <= uk]

    for i, layer in enumerate(layers):
        x_in = x_vars[i]
        x_out = x_vars[i + 1]

        if isinstance(layer, nn.Linear):
            W = to_np(layer.weight)
            b = to_np(layer.bias).reshape(-1)
            cons += [x_out == W @ x_in + b]

        elif isinstance(layer, nn.ReLU):
            lz = to_np(l_bounds[i]).reshape(-1)
            uz = to_np(u_bounds[i]).reshape(-1)
            delta = cp.Variable(dims[i], boolean=True)

            cons += [x_out >= x_in, x_out >= 0]
            cons += [x_out <= x_in - cp.multiply(lz, (1 - delta))]
            cons += [x_out <= cp.multiply(uz, delta)]
        else:
            raise TypeError(f"Unsupported layer in MILP baseline: {type(layer)}")

    c_np = to_np(c).reshape(-1)
    prob = cp.Problem(cp.Maximize(c_np @ x_vars[K] + float(d)), cons)
    val = prob.solve(solver=solver, verbose=verbose)

    return {
        "status": prob.status,
        "opt": float(val),
        "solver": solver,
        "num_cons": len(cons),
        "x0": x_vars[0].value,
        "xK": x_vars[K].value,
    }


# ====================== Random test case ======================

def make_random_mlp(in_dim: int, hidden_dims: List[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def generate_random_test_case(
    in_dim: int = 4,
    hidden_dims: List[int] = [8, 8],
    out_dim: int = 3,
    eps: float = 0.1,
    box_radius: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_random_mlp(in_dim, hidden_dims, out_dim).to(device)
    layers = list(model)

    x_nom = torch.randn(1, in_dim, device=device)

    with torch.no_grad():
        out_nom = model(x_nom)
    out_dim_real = out_nom.shape[-1]

    i, j = torch.randint(0, out_dim_real, (2,))
    while j == i:
        j = torch.randint(0, out_dim_real, (1,)).item()

    c = torch.zeros_like(out_nom, device=device)
    c[0, j] = 1.0
    c[0, i] = -1.0
    d = 0.0

    l0_box = (x_nom - box_radius).clone()
    u0_box = (x_nom + box_radius).clone()

    return {
        "model": model,
        "layers": layers,
        "x_nom": x_nom,
        "c": c,
        "d": d,
        "eps": eps,
        "l0_box": l0_box,
        "u0_box": u0_box,
        "i": int(i),
        "j": int(j),
    }


def run_random_tests(
    num_tests: int = 1,
    in_dim: int = 4,
    hidden_dims: List[int] = [8, 8],
    out_dim: int = 3,
    eps: float = 0.1,
    box_radius: float = 1.0,
    pd_steps: int = 800,
    cvx_lp_solver: str = "ECOS",
    cvx_milp_solver: str = "ECOS_BB",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    for t in range(num_tests):
        print(f"\n===== Test case {t+1}/{num_tests} =====")
        case = generate_random_test_case(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            eps=eps,
            box_radius=box_radius,
            device=device,
        )

        model = case["model"]
        layers = case["layers"]
        x_nom = case["x_nom"]
        c = case["c"]
        d = case["d"]
        l0_box = case["l0_box"]
        u0_box = case["u0_box"]
        i = case["i"]
        j = case["j"]

        print(f"Property: f_{j}(x) - f_{i}(x) <= 0, eps = {eps}")

        # IBP bounds for projections (paper's C_k)
        l_bounds, u_bounds = compute_ibp_bounds(layers, x_nom, eps, l0_box, u0_box)

        K = len(layers)
        # step sizes
        tau_list = [0.05] * (K + 1)
        sigma_list = [0.05] * K

        pd_res = dual_g_lamda(
            layers=layers,
            x_nom=x_nom,
            c=c,
            d=d,
            eps=eps,
            l0_box=l0_box,
            u0_box=u0_box,
            l_bounds=l_bounds,
            u_bounds=u_bounds,
            tau_list=tau_list,
            sigma_list=sigma_list,
            n_steps=pd_steps,
            enforce_last_lambda_eq_minus_c=True,
            strict_eval_oracle_every=20,  # set 0 to disable
        )

        print(f"[Paper-PD] best g_oracle(λ) (strict cert metric): {pd_res['best_g_oracle']:.6f}  Certified: {pd_res['certified_by_oracle']}")

        # ---- CVX-LP triangle relaxation baseline ----
        try:
            lp_res = cvxpy_triangle_lp_baseline(
                layers=layers,
                c=c,
                d=d,
                l_bounds=l_bounds,
                u_bounds=u_bounds,
                solver=cvx_lp_solver,
                verbose=False,
            )
            lp_val = lp_res["opt"]
            print(f"[CVX-LP (triangle)] status={lp_res['status']} opt={lp_val:.6f}  Certified: {lp_val < 0}")
        except Exception as e:
            print(f"[CVX-LP (triangle)] Failed: {repr(e)}")

        # ---- CVX-MILP exact baseline (solves original problem) ----
        try:
            milp_res = cvxpy_milp_exact_baseline(
                layers=layers,
                c=c,
                d=d,
                l_bounds=l_bounds,
                u_bounds=u_bounds,
                solver=cvx_milp_solver,
                verbose=False,
            )
            milp_val = milp_res["opt"]
            print(f"[CVX-MILP exact] solver={milp_res['solver']} status={milp_res['status']} opt={milp_val:.6f}  Certified: {milp_val < 0}")
            print(f"[CVX-MILP exact] (#constraints={milp_res['num_cons']})")

            # optional: verify MILP counterexample in PyTorch
            if milp_res["x0"] is not None:
                x0_star = torch.tensor(milp_res["x0"], dtype=x_nom.dtype, device=x_nom.device).view_as(x_nom)
                with torch.no_grad():
                    out_star = model(x0_star)
                    val_check = float((c.flatten() * out_star.flatten()).sum().item() + d)
                print(f"[MILP check] c^T f(x0*) + d = {val_check:.6f} (should match MILP opt up to tol)")
        except Exception as e:
            print(f"[CVX-MILP exact] Failed: {repr(e)}")
            print("Tip: ECOS_BB may be slow; GUROBI/CPLEX will be faster if available.")


if __name__ == "__main__":
    run_random_tests(
        num_tests=3,
        in_dim=4,
        hidden_dims=[8, 8],
        out_dim=3,
        eps=0.1,
        box_radius=1.0,
        pd_steps=2000,
        cvx_lp_solver="ECOS",
        cvx_milp_solver="ECOS_BB",
    )
