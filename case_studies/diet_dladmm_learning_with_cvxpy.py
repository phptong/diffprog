import torch
from torch import nn
from typing import Optional, Tuple


# -----------------------------
# Helper: estimate spectral norm squared of A (for PGD step size)
# -----------------------------
@torch.no_grad()
def estimate_spectral_norm_sq(A: torch.Tensor, iters: int = 50) -> float:

    m, n = A.shape
    v = torch.randn(n, device=A.device, dtype=A.dtype)
    v = v / (torch.norm(v) + 1e-12)
    for _ in range(iters):
        Av = A @ v
        AtAv = A.t() @ Av
        v = AtAv / (torch.norm(AtAv) + 1e-12)
    # Rayleigh quotient for A^T A
    Av = A @ v
    norm_sq = (Av @ Av).item()
    return float(norm_sq)


# -----------------------------
# Route B ADMM (differentiable unrolled)
# -----------------------------
class SlackADMMRouteB(nn.Module):
    def __init__(
        self,
        max_iters: int = 2000,
        rho: float = 1.0,
        x_inner_iters: int = 50,
        x_step: Optional[float] = None,
        return_history: bool = False,
    ):
        super().__init__()
        self.max_iters = max_iters
        self.rho = rho
        self.x_inner_iters = x_inner_iters
        self.x_step = x_step
        self.return_history = return_history

    def forward(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        s0: Optional[torch.Tensor] = None,
        u0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        assert A.dim() == 2 and b.dim() == 1 and c.dim() == 1
        m, n = A.shape
        assert b.shape[0] == m and c.shape[0] == n

        device, dtype = A.device, A.dtype
        rho = self.rho

        x = torch.zeros(n, device=device, dtype=dtype) if x0 is None else x0
        s = torch.zeros(m, device=device, dtype=dtype) if s0 is None else s0
        u = torch.zeros(m, device=device, dtype=dtype) if u0 is None else u0

        # Step size for inner PGD
        if self.x_step is None:
            x_step = 1e-3
        else:
            x_step = float(self.x_step)

        if self.return_history:
            x_hist = [x.unsqueeze(0)]
        else:
            x_hist = None

        for _ in range(self.max_iters):
            # x-update: approximately solve
            #   min_{x>=0} c^T x + (rho/2)||A x - (s + b - u)||^2
            r = s + b - u  # target for Ax

            for _ in range(self.x_inner_iters):
                # grad = c + rho * A^T(Ax - r)
                grad = c + rho * (A.t() @ (A @ x - r))
                x = x - x_step * grad
                x = torch.clamp(x, min=0.0)  # projection to x>=0

            # s-update: projection to s>=0
            s = torch.clamp(A @ x - b + u, min=0.0)

            # u-update
            u = u + (A @ x - s - b)

            if self.return_history:
                x_hist.append(x.unsqueeze(0))

        # Map to original KKT duals
        p = rho * u
        lambda_k = torch.clamp(-p, min=0.0)               # for Ax >= b
        nu_k = torch.clamp(c + A.t() @ p, min=0.0)        # for x >= 0

        if self.return_history:
            x_hist = torch.cat(x_hist, dim=0)

        return x, s, u, lambda_k, nu_k, x_hist


# -----------------------------
# CVXPY baseline (primal + dual)
# -----------------------------
def solve_with_cvxpy(A: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """
    min c^T x
    s.t. A x >= b
         x >= 0
    Returns:
        x_cvx, obj, status, solver, lambda_cvx, nu_cvx
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is not installed. Please install it first: pip install cvxpy\n"
            "Also, ensure that a solver is available (such as ECOS/SCS/OSQP)."
        ) from e

    A_np = A.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    c_np = c.detach().cpu().numpy()

    m, n = A_np.shape
    x = cp.Variable(n)

    constr_ax = (A_np @ x >= b_np)
    constr_xn = (x >= 0)

    prob = cp.Problem(cp.Minimize(c_np @ x), [constr_ax, constr_xn])

    solver_candidates = ["ECOS", "OSQP", "SCS", "CLARABEL", "SCIPY"]
    last_err = None
    used_solver = None
    for s in solver_candidates:
        try:
            prob.solve(solver=s, verbose=False)
            used_solver = s
            break
        except Exception as err:
            last_err = err

    if used_solver is None:
        raise RuntimeError(f"cvxpy failed to solve. Last error: {last_err}")
    if x.value is None:
        raise RuntimeError(f"cvxpy completed the calculation but did not return a solution (status={prob.status}).")

    x_val = x.value
    x_cvx = torch.tensor(x_val, dtype=A.dtype)

    obj = float(c_np @ x_val)
    status = prob.status

    lam_raw = constr_ax.dual_value  # >=0
    nu_raw = constr_xn.dual_value   # >=0
    if lam_raw is None or nu_raw is None:
        raise RuntimeError("cvxpy does not return a dual variable (dual_value=None).")

    lam_cvx = torch.tensor(lam_raw, dtype=A.dtype)
    nu_cvx = torch.tensor(nu_raw, dtype=A.dtype)

    return x_cvx, obj, status, used_solver, lam_cvx, nu_cvx


# -----------------------------
# Compare & report
# -----------------------------
@torch.no_grad()
def report_compare(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    x_admm: torch.Tensor,
    lambda_admm: torch.Tensor,
    nu_admm: torch.Tensor,
    tag: str = "",
):
    x_cvx, obj_cvx, status, solver, lam_cvx, nu_cvx = solve_with_cvxpy(A, b, c)

    obj_admm = (c @ x_admm).item()
    Ax_admm = A @ x_admm
    Ax_cvx = A @ x_cvx

    viol_admm = torch.clamp(b - Ax_admm, min=0.0)
    viol_cvx = torch.clamp(b - Ax_cvx, min=0.0)

    print("\n" + "=" * 70)
    print(f"CVXPY baseline compare {('[' + tag + ']') if tag else ''}")
    print("=" * 70)
    print(f"CVXPY: status={status}, solver={solver}")
    print(f"objective: admm={obj_admm:.8f} | cvx={obj_cvx:.8f} | |diff|={abs(obj_admm-obj_cvx):.3e}")
    print(f"feas violation max: admm={viol_admm.max().item():.3e} | cvx={viol_cvx.max().item():.3e}")
    print(f"feas violation L1 : admm={viol_admm.sum().item():.3e} | cvx={viol_cvx.sum().item():.3e}")
    print(f"||x_admm - x_cvx||2: {torch.norm(x_admm - x_cvx).item():.3e}")

    print(f"||lambda_admm - lambda_cvx||2: {torch.norm(lambda_admm - lam_cvx).item():.3e}")
    print(f"||nu_admm     - nu_cvx||2:     {torch.norm(nu_admm - nu_cvx).item():.3e}")

    # Complementarity
    comp_admm_lam = lambda_admm * (b - Ax_admm)
    comp_cvx_lam = lam_cvx * (b - Ax_cvx)

    comp_admm_nu = nu_admm * x_admm
    comp_cvx_nu = nu_cvx * x_cvx

    print("\nComplementarity norms (smaller is better)")
    print(f"ADMM: ||λ⊙(b-Ax)||2={torch.norm(comp_admm_lam).item():.3e} | ||ν⊙x||2={torch.norm(comp_admm_nu).item():.3e}")
    print(f"CVX : ||λ⊙(b-Ax)||2={torch.norm(comp_cvx_lam).item():.3e} | ||ν⊙x||2={torch.norm(comp_cvx_nu).item():.3e}")

    print("\nDuals snapshot")
    print("lambda_admm:", lambda_admm.detach().cpu().numpy())
    print("lambda_cvx :", lam_cvx.detach().cpu().numpy())
    print("nu_admm    :", nu_admm.detach().cpu().numpy())
    print("nu_cvx     :", nu_cvx.detach().cpu().numpy())


# -----------------------------
# Demo: toy problem + learning c + baseline check
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    n = 9
    m = 4

    A_true = torch.rand(m, n) * 10.0
    b_true = torch.rand(m) * 20.0 + 5.0
    c_true = torch.rand(n) * 2.0 + 0.5

    # (Optional but recommended) set a stable PGD step size based on ||A||_2^2
    # x_step ≈ 1 / (rho * ||A||^2). Use a safety factor.
    rho = 1.0
    norm_sq = estimate_spectral_norm_sq(A_true, iters=60)
    x_step = 0.9 / (rho * norm_sq + 1e-12)

    # Learnable c
    c_param = nn.Parameter(c_true.clone())

    # Route-B ADMM layer
    solver = SlackADMMRouteB(
        max_iters=1000,
        rho=rho,
        x_inner_iters=30,
        x_step=x_step,
        return_history=False,
    )

    # Target solution from true c (no gradient needed here)
    with torch.no_grad():
        x_star, s_star, u_star, lam_star, nu_star, _ = solver(A_true, b_true, c_true)

    # Baseline check: ADMM vs CVXPY (true c)
    with torch.no_grad():
        report_compare(A_true, b_true, c_true, x_star, lam_star, nu_star, tag="using true c (RouteB ADMM vs CVX)")

    optimizer = torch.optim.Adam([c_param], lr=1e-2)

    for step in range(50):
        optimizer.zero_grad()

        x_pred, s_pred, u_pred, lam_pred, nu_pred, _ = solver(A_true, b_true, c_param)

        loss = torch.nn.functional.mse_loss(x_pred, x_star)
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1:02d} | Loss = {loss.item():.6f}")

    print("\nTrue c:", c_true.detach().cpu().numpy())
    print("Learned c:", c_param.detach().cpu().numpy())
    print("Solution x (with learned c):", x_pred.detach().cpu().numpy())

    # Baseline check: ADMM vs CVXPY (learned c)
    with torch.no_grad():
        report_compare(A_true, b_true, c_param.detach(), x_pred, lam_pred, nu_pred, tag="using learned c (RouteB ADMM vs CVX)")
