import numpy as np
import torch
import cvxpy as cp
from typing import Dict, Tuple


# ============================================================
# Baseline helpers (CVXPY + metrics)
# ============================================================

def objective_np(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    r = A @ x - b
    return 0.5 * float(r @ r)

def grad_np(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return A.T @ (A @ x - b)

def kkt_residual_nnls_np(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:

    g = grad_np(A, b, x)
    mu = np.maximum(0.0, g)
    r_stationarity = np.linalg.norm(g - mu)
    r_complementarity = np.linalg.norm(x * mu)
    r_primal = np.linalg.norm(np.minimum(x, 0.0))
    return float(np.sqrt(r_stationarity**2 + r_complementarity**2 + r_primal**2))

def solve_nnls_cvxpy_baseline(
    A: np.ndarray,
    b: np.ndarray,
    solver_preference: Tuple[str, ...] = ("OSQP", "ECOS", "SCS"),
) -> Tuple[np.ndarray, Dict[str, object]]:

    m, n = A.shape
    x = cp.Variable(n)
    constraints = [x >= 0]
    obj = 0.5 * cp.sum_squares(A @ x - b)
    prob = cp.Problem(cp.Minimize(obj), constraints)

    last_err = None
    for s in solver_preference:
        try:
            prob.solve(solver=s, verbose=False)
            if x.value is not None:
                break
        except Exception as e:
            last_err = e

    if x.value is None:
        raise RuntimeError(f"CVXPY failed to solve NNLS. Last error: {last_err}")

    x_star = np.asarray(x.value).reshape(-1)

    # Dual for constraint x >= 0
    mu_dual = constraints[0].dual_value
    if mu_dual is not None:
        mu_dual = np.asarray(mu_dual).reshape(-1)
    # KKT-reconstructed mu (solver-agnostic)
    mu_kkt = np.maximum(0.0, A.T @ (A @ x_star - b))

    diag = {
        "status": prob.status,
        "obj": float(prob.value),
        "kkt_primal_only": kkt_residual_nnls_np(A, b, x_star),
        "mu_dual": mu_dual,      # may be None if solver doesn't provide
        "mu_kkt": mu_kkt,        # always available
        "mu_dual_minus_mu_kkt_norm": (
            float(np.linalg.norm(mu_dual - mu_kkt)) if mu_dual is not None else None
        ),
        "mu_dual_min": (float(mu_dual.min()) if mu_dual is not None else None),
        "mu_dual_max": (float(mu_dual.max()) if mu_dual is not None else None),
    }
    return x_star, diag


# ============================================================
# Dual ADMM (scaled) for the dual-min splitting form
# ============================================================

def nnls_dual_admm(
    A: torch.Tensor,
    b: torch.Tensor,
    rho: float = 1.0,
    max_iter: int = 5000,
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:


    assert A.ndim == 2
    assert b.ndim == 1 and b.shape[0] == A.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = A.to(device)
    b = b.to(device)

    m, n = A.shape
    AT = A.T

    # (I + rho A A^T) is SPD -> cholesky
    I_m = torch.eye(m, dtype=A.dtype, device=A.device)
    M = I_m + rho * (A @ AT)
    L = torch.linalg.cholesky(M)

    def solve_M(rhs: torch.Tensor) -> torch.Tensor:
        return torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)

    lam = torch.zeros(m, dtype=A.dtype, device=A.device)
    mu  = torch.zeros(n, dtype=A.dtype, device=A.device)
    v   = torch.zeros(n, dtype=A.dtype, device=A.device)

    r_norm = float("inf")
    s_norm = float("inf")

    for k in range(max_iter):
        # lam-update
        rhs = -b + rho * (A @ (mu - v))
        lam = solve_M(rhs)

        mu_old = mu.clone()

        # mu-update (projection)
        mu = torch.clamp(AT @ lam + v, min=0.0)

        # v-update
        v = v + (AT @ lam - mu)

        # residuals (standard)
        r = AT @ lam - mu
        s = rho * (mu - mu_old)
        r_norm = torch.linalg.norm(r).item()
        s_norm = torch.linalg.norm(s).item()

        # thresholds
        a = torch.linalg.norm(AT @ lam)
        c = torch.linalg.norm(mu)
        e_pri = (n ** 0.5) * eps_abs + eps_rel * torch.max(a, c)
        e_dual = (n ** 0.5) * eps_abs + eps_rel * torch.linalg.norm(rho * v)

        if r_norm <= float(e_pri) and s_norm <= float(e_dual):
            if verbose:
                print(f"Converged at iter {k+1} (r={r_norm:.3e}, s={s_norm:.3e}).")
            break
    else:
        if verbose:
            print(f"Did not converge within {max_iter} iters (r={r_norm:.3e}, s={s_norm:.3e}).")

    return {
        "lam": lam,
        "mu": mu,
        "v": v,
        "r_norm": torch.tensor(r_norm, device=device),
        "s_norm": torch.tensor(s_norm, device=device),
        "iters": torch.tensor(k + 1, device=device),
    }


# ============================================================
# Primal recovery from accurate mu via active-set refinement
# ============================================================

def recover_x_from_mu_active_set(
    A: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    tau: float = 1e-10,
    max_refine: int = 100,
) -> np.ndarray:

    m, n = A.shape
    mu = mu.reshape(-1)
    assert mu.shape[0] == n

    # initial free set: where mu ~ 0
    F = (mu <= tau)
    x = np.zeros(n)

    for _ in range(max_refine):
        idx = np.where(F)[0]
        if idx.size == 0:
            return np.zeros(n)

        A_F = A[:, idx]
        x_F, *_ = np.linalg.lstsq(A_F, b, rcond=None)

        if np.all(x_F >= 0):
            x[:] = 0.0
            x[idx] = x_F
            return x

        # remove negative components from free set
        bad = idx[x_F < 0]
        if bad.size == 0:
            x[:] = 0.0
            x[idx] = np.maximum(x_F, 0.0)
            return x
        F[bad] = False

    # fallback
    idx = np.where(F)[0]
    x[:] = 0.0
    if idx.size > 0:
        x_F, *_ = np.linalg.lstsq(A[:, idx], b, rcond=None)
        x[idx] = np.maximum(x_F, 0.0)
    return x


# ============================================================
# Dual comparison helper (against CVXPY primal KKT reconstruction)
# ============================================================

def dual_comparison_report(A: np.ndarray, b: np.ndarray, x_star: np.ndarray,mu_star,
                           lam_admm: np.ndarray, mu_admm: np.ndarray) -> None:

    x_star = x_star.reshape(-1)

    lam_cvx_std = A @ x_star - b
    mu_cvx = np.maximum(0.0, A.T @ (A @ x_star - b))

    lam_admm = lam_admm.reshape(-1)
    mu_admm = mu_admm.reshape(-1)

    # sign alignment for lam
    err_plus = np.linalg.norm(lam_admm - lam_cvx_std)
    err_minus = np.linalg.norm(lam_admm + lam_cvx_std)
    lam_sign = +1.0 if err_plus <= err_minus else -1.0
    lam_admm_aligned = lam_sign * lam_admm

    print("\n[Dual Comparison]")
    print("  (A) Compare mu (n-dim, multiplier for x>=0):")
    print(f"    ||mu_admm - mu_cvx||_2 = {np.linalg.norm(mu_admm - mu_star):.6e}")
    print(f"    ||mu_cvx||_2           = {np.linalg.norm(mu_star):.6e}")
    print(f"    rel_err(mu)            = {np.linalg.norm(mu_admm - mu_cvx)/(np.linalg.norm(mu_cvx)+1e-12):.6e}")
    print(f"    mu_admm min/max        = ({mu_admm.min():.3e}, {mu_admm.max():.3e})")
    print(f"    mu_cvx  min/max        = ({mu_cvx.min():.3e}, {mu_cvx.max():.3e})")

    print("\n  (B) Compare lambda (m-dim, residual-type dual), with sign alignment:")
    print(f"    chosen sign for lam_admm = {lam_sign:+.0f}")
    print(f"    ||lam_admm_aligned - (Ax*-b)||_2 = {np.linalg.norm(lam_admm_aligned - lam_cvx_std):.6e}")
    print(f"    ||Ax*-b||_2                    = {np.linalg.norm(lam_cvx_std):.6e}")
    print(f"    rel_err(lam)                   = {np.linalg.norm(lam_admm_aligned - lam_cvx_std)/(np.linalg.norm(lam_cvx_std)+1e-12):.6e}")

    print("\n  (C) Internal consistency checks:")
    print(f"    ||A^T lam_admm_aligned - mu_admm||_2 = {np.linalg.norm(A.T @ lam_admm_aligned - mu_admm):.6e}")
    print(f"    min(mu_admm) = {mu_admm.min():.3e}")


# ============================================================
# Main demo
# ============================================================

if __name__ == "__main__":
    # ----- Generate a random NNLS instance -----
    torch.manual_seed(0)
    np.random.seed(0)

    m, n = 80, 50
    A_t = torch.randn(m, n, dtype=torch.float64)
    x_true = torch.rand(n, dtype=torch.float64)
    x_true[x_true < 0.8] = 0.0
    b_t = A_t @ x_true + 0.01 * torch.randn(m, dtype=torch.float64)

    # numpy copies
    A_np = A_t.detach().cpu().numpy()
    b_np = b_t.detach().cpu().numpy()

    # ----- CVXPY baseline -----
    x_star,cvx_diag = solve_nnls_cvxpy_baseline(A_np, b_np)
    mu_star = cvx_diag['mu_dual']
    print("\n[CVXPY Baseline]")
    print(f"  status = {cvx_diag['status']}")
    print(f"  obj(x_star) = {0.5*np.linalg.norm(A_np @ x_star - b_np):.6e}")
    print(f"  KKT_primal_only(x_star) = {cvx_diag['kkt_primal_only']:.3e}")

    # ----- Dual ADMM -----
    out = nnls_dual_admm(
        A_t, b_t,
        rho=1.0,
        max_iter=20000,
        eps_abs=1e-8,
        eps_rel=1e-8,
        verbose=True,
    )

    lam_admm = out["lam"].detach().cpu().numpy()
    mu_admm = out["mu"].detach().cpu().numpy()

    # ----- Primal recovery from mu (active-set refinement) -----
    tau = 1e-10
    x_rec = recover_x_from_mu_active_set(A_np, b_np, mu_admm, tau=tau, max_refine=200)

    print("\n[Dual-ADMM + Primal Recovery]")
    print(f"  tau = {tau:.1e}")
    print(f"  obj(x_rec) = {0.5*np.linalg.norm(A_np @ x_rec - b_np):.6e}")
    print(f"  ||x_rec - x_star||_2 = {np.linalg.norm(x_rec - x_star):.6e}")
    print(f"  KKT_primal_only(x_rec) = {kkt_residual_nnls_np(A_np, b_np, x_rec):.3e}")
    print(f"  iters = {int(out['iters'].item())}")
    print(f"  recovered nnz(x_rec) = {int(np.sum(x_rec > 0))}")

    # ----- Dual comparison -----
    dual_comparison_report(A_np, b_np, x_star,mu_star, lam_admm, mu_admm)
