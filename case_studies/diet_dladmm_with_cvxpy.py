import torch
from torch import nn
from typing import Optional, Tuple


# -----------------------------
# Route B ADMM solver
# -----------------------------
class ADMMStiglerRouteB(nn.Module):
    def __init__(
        self,
        max_iters: int = 2000,
        rho: float = 1.0,
        x_inner_iters: int = 50,
        x_inner_step: Optional[float] = None,
        return_history: bool = False,
    ):
        """
        max_iters: ADMM outer iterations
        rho: ADMM penalty parameter
        x_inner_iters: number of PGD steps to approximately solve x-subproblem
        x_inner_step: PGD step size; if None uses 1/L with L ~ rho * ||A||_2^2
        return_history: save x history
        """
        super().__init__()
        self.max_iters = max_iters
        self.rho = rho
        self.x_inner_iters = x_inner_iters
        self.x_inner_step = x_inner_step
        self.return_history = return_history

    @torch.no_grad()
    def _x_update_pgd(
        self,
        A: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor,
        b: torch.Tensor,
        u: torch.Tensor,
        x_init: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve:
            min_{x>=0}  c^T x + (rho/2)||A x - s - b + u||^2
        via projected gradient descent (PGD).
        """
        rho = self.rho
        # r = s + b - u  -> term is ||A x - r||^2 up to sign:
        # Actually ||A x - s - b + u||^2 = ||A x - (s + b - u)||^2
        r = s + b - u

        # Lipschitz constant of grad: rho * ||A||_2^2
        if self.x_inner_step is None:
            # spectral norm estimate via svd for small sizes (safe for your n=9)
            # if larger, you'd use power iteration.
            svals = torch.linalg.svdvals(A)
            L = rho * (svals.max().item() ** 2 + 1e-12)
            step = 1.0 / L
        else:
            step = self.x_inner_step

        x = x_init.clone()

        for _ in range(self.x_inner_iters):
            # grad = c + rho * A^T (A x - r)
            grad = c + rho * (A.t() @ (A @ x - r))
            x = x - step * grad
            x = torch.clamp(x, min=0.0)  # projection onto x>=0

        return x

    def forward(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        s0: Optional[torch.Tensor] = None,
        u0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            x: primal decision
            s: slack (Ax - s = b, s>=0)
            u: scaled equality dual
            lambda_kkt: KKT lambda for Ax>=b
            x_hist: optional history
        """
        assert A.dim() == 2 and b.dim() == 1 and c.dim() == 1
        m, n = A.shape
        assert b.shape[0] == m and c.shape[0] == n

        device, dtype = A.device, A.dtype

        x = torch.zeros(n, device=device, dtype=dtype) if x0 is None else x0.clone()
        s = torch.zeros(m, device=device, dtype=dtype) if s0 is None else s0.clone()
        u = torch.zeros(m, device=device, dtype=dtype) if u0 is None else u0.clone()

        if self.return_history:
            x_hist = [x.unsqueeze(0)]
        else:
            x_hist = None

        for _ in range(self.max_iters):
            # x-update (approx) with PGD
            x = self._x_update_pgd(A, c, s, b, u, x_init=x)

            # s-update: projection onto s>=0
            s = torch.clamp(A @ x - b + u, min=0.0)

            # u-update (scaled dual)
            u = u + (A @ x - s - b)

            if self.return_history:
                x_hist.append(x.unsqueeze(0))

        # KKT-mapped lambda (for Ax>=b): lambda = -rho*u (should be >=0)
        lambda_kkt = -self.rho * u

        if self.return_history:
            x_hist = torch.cat(x_hist, dim=0)

        return x, s, u, lambda_kkt, x_hist


# -----------------------------
# CVXPY baseline with duals
# -----------------------------
def solve_with_cvxpy(A: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """
    min c^T x
    s.t. A x >= b
         x >= 0
    return x, obj, status, solver, lambda(Ax>=b), nu(x>=0)
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("cvxpy is not installed: pip install cvxpy") from e

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
        raise RuntimeError(f"cvxpy failed, final error:{last_err}")

    x_val = x.value
    if x_val is None:
        raise RuntimeError(f"cvxpy status={prob.status} no solution")

    x_cvx = torch.tensor(x_val, dtype=A.dtype)
    obj = float(c_np @ x_val)

    lam_raw = constr_ax.dual_value  # should be >=0 for >= constraint in cvxpy convention
    nu_raw = constr_xn.dual_value   # should be >=0

    if lam_raw is None or nu_raw is None:
        raise RuntimeError("cvxpy dual_value=None")

    lam_cvx = torch.tensor(lam_raw, dtype=A.dtype)
    nu_cvx = torch.tensor(nu_raw, dtype=A.dtype)

    return x_cvx, obj, prob.status, used_solver, lam_cvx, nu_cvx


# -----------------------------
# Demo / comparison
# -----------------------------
def main():
    torch.manual_seed(0)

    n = 9
    m = 4
    A = torch.rand(m, n) * 10.0
    b = torch.rand(m) * 20.0 + 5.0
    c = torch.rand(n) * 2.0 + 0.5

    # --- Route B ADMM ---
    admm = ADMMStiglerRouteB(
        max_iters=4000,
        rho=1.0,
        x_inner_iters=60,
        x_inner_step=None,
        return_history=False,
    )
    x_admm, s_admm, u_admm, lam_admm, _ = admm(A, b, c)

    # KKT nu from ADMM dual (this is now "true" KKT nu under the slack KKT mapping)
    # p = rho*u, nu = c + A^T p
    nu_admm = torch.clamp(c + admm.rho * (A.t() @ u_admm), min=0.0)

    obj_admm = (c @ x_admm).item()
    Ax_admm = A @ x_admm
    viol_admm = torch.clamp(b - Ax_admm, min=0.0)

    # --- CVXPY ---
    x_cvx, obj_cvx, status, solver, lam_cvx, nu_cvx = solve_with_cvxpy(A, b, c)
    Ax_cvx = A @ x_cvx
    viol_cvx = torch.clamp(b - Ax_cvx, min=0.0)

    # --- prints ---
    print("=== Stigler Diet: Route-B ADMM vs CVXPY baseline ===\n")
    print("A shape:", tuple(A.shape))
    print("b:", b.detach().cpu().numpy())
    print("c:", c.detach().cpu().numpy())

    print("\n---------------------")
    print("Route-B ADMM result")
    print("---------------------")
    print("x_admm:")
    print(x_admm.detach().cpu().numpy())
    print(f"objective c^T x_admm: {obj_admm:.8f}")
    print("A x_admm:", Ax_admm.detach().cpu().numpy())
    print("feasibility (A x_admm - b):", (Ax_admm - b).detach().cpu().numpy())
    print(f"constraint violation: max={viol_admm.max().item():.6e}, L1={viol_admm.sum().item():.6e}")

    # lam_admm ideally >=0; if small negatives appear due to inexactness, clamp for reporting
    lam_admm_report = torch.clamp(lam_admm, min=0.0)

    print("\nADMM KKT duals (mapped)")
    print("lambda_admm (for Ax >= b):")
    print(lam_admm_report.detach().cpu().numpy())
    print("nu_admm (for x >= 0):")
    print(nu_admm.detach().cpu().numpy())

    # Complementarity checks (KKT-style)
    comp_lam = lam_admm_report * (b - Ax_admm)
    comp_nu = nu_admm * x_admm
    print("\nADMM complementarity")
    print("λ ⊙ (b - Ax):", comp_lam)
    print("Norm of λ ⊙ (b - Ax):", torch.norm(comp_lam).item())
    print("ν ⊙ x:", comp_nu)
    print("Norm of ν ⊙ x:", torch.norm(comp_nu).item())

    # Stationarity residual: c - A^T λ - ν should be ~0
    stat_res = c - A.t() @ lam_admm_report - nu_admm
    print("\nADMM stationarity residual ||c - A^T λ - ν||2:", torch.norm(stat_res).item())

    print("\n---------------------")
    print("CVXPY baseline result")
    print("---------------------")
    print(f"status: {status}, solver: {solver}")
    print("x_cvx:")
    print(x_cvx.detach().cpu().numpy())
    print(f"objective c^T x_cvx:  {obj_cvx:.8f}")
    print("A x_cvx:", Ax_cvx.detach().cpu().numpy())
    print("feasibility (A x_cvx - b):", (Ax_cvx - b).detach().cpu().numpy())
    print(f"constraint violation: max={viol_cvx.max().item():.6e}, L1={viol_cvx.sum().item():.6e}")

    print("\nCVXPY duals")
    print("lambda_cvx (for Ax >= b):")
    print(lam_cvx.detach().cpu().numpy())
    print("nu_cvx (for x >= 0):")
    print(nu_cvx.detach().cpu().numpy())

    # --- comparisons ---
    print("\n---------------------")
    print("Comparison")
    print("---------------------")
    print(f"|obj_admm - obj_cvx|: {abs(obj_admm - obj_cvx):.8e}")
    print(f"||x_admm - x_cvx||2: {torch.norm(x_admm - x_cvx).item():.8e}")

    print("\nDual comparison")
    print(f"||lambda_admm - lambda_cvx||2: {torch.norm(lam_admm_report - lam_cvx).item():.8e}")
    print(f"||nu_admm     - nu_cvx||2:     {torch.norm(nu_admm - nu_cvx).item():.8e}")


if __name__ == "__main__":
    main()
