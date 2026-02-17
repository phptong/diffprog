import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# 1. PDHG / Chambolle–Pock solver
# ============================
class StiglerPDHGSolver(nn.Module):


    def __init__(self, n_iters=2000, tau=None, sigma=None, theta=1.0):
        super().__init__()
        self.n_iters = n_iters
        self.tau = tau
        self.sigma = sigma
        self.theta = theta

    def _compute_stepsizes(self, A):

        with torch.no_grad():
            svals = torch.linalg.svdvals(A)
            L = svals.max().item()  # ≈ ||A||_2
            L2 = L * L + 1e-6

            tau = 0.9 / L2
            sigma = 0.9 / L2
        return tau, sigma

    def prox_f(self, v, tau, c):

        return F.relu(v - tau * c)

    def prox_g_conj(self, y, sigma, b):

        return torch.minimum(torch.zeros_like(y), y - sigma * b)

    def forward(self, c, A, b, x0=None, y0=None):

        device = c.device
        dtype = c.dtype
        m, n = A.shape

        if self.tau is None or self.sigma is None:
            tau, sigma = self._compute_stepsizes(A)
        else:
            tau, sigma = self.tau, self.sigma

        if x0 is None:
            x = torch.zeros(n, device=device, dtype=dtype)
        else:
            x = x0

        if y0 is None:
            y = torch.zeros(m, device=device, dtype=dtype)
        else:
            y = y0

        x_bar = x.clone()
        theta = self.theta

        for _ in range(self.n_iters):
            # y update: y^{k+1} = prox_{σ g*}(y^k + σ A \bar x^k)
            y = self.prox_g_conj(y + sigma * A.matmul(x_bar), sigma, b)

            # x update: x^{k+1} = prox_{τ f}(x^k - τ A^T y^{k+1})
            x_old = x
            x = self.prox_f(x - tau * A.t().matmul(y), tau, c)

            # extrapolation
            x_bar = x + theta * (x - x_old)


        lam = F.relu(-y)

        # stationarity: c - A^T λ - ν = 0 => ν = c - A^T λ
        nu_raw = c - A.t().matmul(lam)
        nu = F.relu(nu_raw)

        return x, lam, nu


# ============================
# 2. Verification
# ============================
def validate_stigler_solution(c, A, b, x, lam, nu, verbose=True):
    with torch.no_grad():
        Ax = A.matmul(x)

        primal_violation_nutrient = F.relu(b - Ax).max().item()
        primal_violation_nonneg = F.relu(-x).max().item()

        dual_violation_lam = F.relu(-lam).max().item()
        dual_violation_nu = F.relu(-nu).max().item()
        dual_residual = (A.t().matmul(lam) + nu - c).abs().max().item()

        comp_lam = (lam * (b - Ax)).abs().max().item()
        comp_nu = (nu * x).abs().max().item()

        primal_obj = torch.dot(c, x).item()
        dual_obj = torch.dot(b, lam).item()
        duality_gap = primal_obj - dual_obj

    if verbose:
        print("===== Stigler PDHG Solution Validation =====")
        print("Primal feasibility:")
        print(f"  Max nutrient violation  max(ReLU(b - A x))  : {primal_violation_nutrient:.4e}")
        print(f"  Max nonneg violation    max(ReLU(-x))      : {primal_violation_nonneg:.4e}")
        print()
        print("Dual feasibility:")
        print(f"  Max λ < 0 violation     max(ReLU(-λ))      : {dual_violation_lam:.4e}")
        print(f"  Max ν < 0 violation     max(ReLU(-ν))      : {dual_violation_nu:.4e}")
        print(f"  Max dual residual       ||A^T λ + ν - c||∞ : {dual_residual:.4e}")
        print()
        print("Complementarity (KKT):")
        print(f"  max |λ_i * (b_i - (A x)_i)| : {comp_lam:.4e}")
        print(f"  max |ν_j * x_j|             : {comp_nu:.4e}")
        print()
        print("Objectives and duality gap:")
        print(f"  Primal objective c^T x : {primal_obj:.6f}")
        print(f"  Dual   objective b^T λ : {dual_obj:.6f}")
        print(f"  Duality gap (P - D)    : {duality_gap:.6e}")
        print("=============================================")

    metrics = {
        "primal_violation_nutrient": primal_violation_nutrient,
        "primal_violation_nonneg": primal_violation_nonneg,
        "dual_violation_lam": dual_violation_lam,
        "dual_violation_nu": dual_violation_nu,
        "dual_residual": dual_residual,
        "comp_lam": comp_lam,
        "comp_nu": comp_nu,
        "primal_obj": primal_obj,
        "dual_obj": dual_obj,
        "duality_gap": duality_gap,
    }
    return metrics


# ============================
# 3. CVXPY baseline
# ============================
def compare_with_cvxpy(c, A, b):
    try:
        import cvxpy as cp
        import numpy as np
    except ImportError:
        print("cvxpy or numpy is not installed, skip the comparison.")
        return None

    A_np = A.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    c_np = c.detach().cpu().numpy()

    n = c_np.shape[0]

    x_var = cp.Variable(n, nonneg=True)
    constraints = [A_np @ x_var >= b_np]
    objective = cp.Minimize(c_np @ x_var)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    x_cvx = x_var.value
    obj_cvx = prob.value

    print("\n===== CVXPY Reference Solution =====")
    print("CVXPY optimal objective:", obj_cvx)
    print("CVXPY optimal x:", x_cvx)
    print("====================================")

    return x_cvx, obj_cvx


# ============================
# 4. Demo
# ============================
if __name__ == "__main__":
    torch.manual_seed(0)

    n = 5
    m = 3

    A = torch.rand(m, n) * 10.0
    b = torch.rand(m) * 20.0 + 5.0
    c = torch.rand(n) * 2.0 + 0.1

    A = A.requires_grad_()
    b = b.requires_grad_()
    c = c.requires_grad_()


    solver = StiglerPDHGSolver(n_iters=150000, tau=None, sigma=None, theta=1)
    x_star, lam_star, nu_star = solver(c, A, b)

    metrics = validate_stigler_solution(c, A, b, x_star, lam_star, nu_star, verbose=True)

    x_cvx, obj_cvx = compare_with_cvxpy(c, A, b)
    print("Torch primal obj:", metrics["primal_obj"])

    slack = A.matmul(x_star) - b
    penalty = F.relu(-slack).sum()
    total_cost = torch.dot(c, x_star)
    loss = total_cost + 10.0 * penalty
    loss.backward()
    #
    # print("\nBackward done.")
    # print("dLoss/dc:", c.grad)
    # print("dLoss/dA:", A.grad)
    # print("dLoss/db:", b.grad)
