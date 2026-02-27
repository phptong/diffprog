import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import cvxpy as cp


# --------------------- Select device ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

script_name = os.path.splitext(os.path.basename(__file__))[0]
script_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(script_dir, script_name)
os.makedirs(fig_dir, exist_ok=True)


# ==========================================================
# 0) CVXPY baseline (ground-truth optimal primal)
# ==========================================================
def solve_nnls_cvxpy(A: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, float]:
    A_np = A.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()

    m, n = A_np.shape
    x = cp.Variable(n, nonneg=True)
    obj = 0.5 * cp.sum_squares(A_np @ x - b_np)
    constraints = [x >= 0]
    prob = cp.Problem(cp.Minimize(obj),constraints)

    # Try OSQP first; if it fails, fall back
    last_err = None
    for s in [cp.OSQP, cp.ECOS, cp.SCS]:
        try:
            prob.solve(solver=s, verbose=False)
            if x.value is not None:
                break
        except Exception as e:
            last_err = e

    if x.value is None:
        raise RuntimeError(f"CVXPY failed to solve NNLS. Last error: {last_err}")

    x_star = torch.from_numpy(np.asarray(x.value).reshape(-1)).to(A.device, dtype=A.dtype)
    mu_dual = constraints[0].dual_value
    if mu_dual is not None:
        mu_dual = np.maximum(0.0, A.T @ (A @ x_star - b))
    # KKT-reconstructed mu (solver-agnostic)


    obj_norm = torch.linalg.norm(A @ x_star - b).item()  # ||Ax-b||_2
    return x_star, obj_norm,mu_dual


# ==========================================================
# 0.5) Active-set recovery from mu (as provided)
# ==========================================================
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


# ==========================================================
# 1) Dual ADMM: solve dual, return mu and recover x via active-set on mu
# ==========================================================
def nnls_dual_admm(
    A: torch.Tensor,
    b: torch.Tensor,
    rho: float = 1.0,
    max_iter: int = 5000,
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
    verbose: bool = False,
    tau_recover: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    assert A.ndim == 2
    assert b.ndim == 1 and b.shape[0] == A.shape[0]

    A = A.to(device)
    b = b.to(device)
    m, n = A.shape
    AT = A.T

    # factor (I + rho A A^T)
    I_m = torch.eye(m, dtype=A.dtype, device=A.device)
    M = I_m + rho * (A @ AT)
    L_chol = torch.linalg.cholesky(M)

    def solve_M(rhs: torch.Tensor) -> torch.Tensor:
        return torch.cholesky_solve(rhs.unsqueeze(1), L_chol).squeeze(1)

    lam = torch.zeros(m, dtype=A.dtype, device=A.device)
    mu  = torch.zeros(n, dtype=A.dtype, device=A.device)
    v   = torch.zeros(n, dtype=A.dtype, device=A.device)

    # (optional) record some primal KKT proxies using recovered x occasionally
    kkt_primal_hist: List[float] = []
    kkt_station_hist: List[float] = []
    kkt_compl_hist: List[float] = []

    r_norm = float("inf")
    s_norm = float("inf")

    for k in range(max_iter):
        # lam-update
        rhs = -b + rho * (A @ (mu - v))
        lam = solve_M(rhs)

        mu_old = mu.clone()

        # mu-update
        mu = torch.clamp(AT @ lam + v, min=0.0)

        # v-update
        v = v + (AT @ lam - mu)

        # residuals (standard n-dim)
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
                print(f"[ADMM] Converged at iter {k+1} (r={r_norm:.3e}, s={s_norm:.3e}).")
            break

    # ---- Recover primal x from mu via active-set ----
    A_np = A.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    x_np = recover_x_from_mu_active_set(A_np, b_np, mu_np, tau=tau_recover, max_refine=200)
    x = torch.from_numpy(x_np).to(device=A.device, dtype=A.dtype)

    # compute KKT residual components for primal NNLS
    with torch.no_grad():
        grad = AT @ (A @ x - b)  # should be >=0
        primal_violation = torch.clamp(-x, min=0.0)
        dual_violation   = torch.clamp(-grad, min=0.0)
        compl_violation  = x * grad
        kkt_primal_hist.append(torch.linalg.norm(primal_violation).item())
        kkt_station_hist.append(torch.linalg.norm(dual_violation).item())
        kkt_compl_hist.append(torch.linalg.norm(compl_violation).item())

    return {
        "x": x.detach().cpu(),
        "lam": lam.detach().cpu(),
        "mu": mu.detach().cpu(),
        "v": v.detach().cpu(),
        "r_norm": torch.tensor(r_norm).cpu(),
        "s_norm": torch.tensor(s_norm).cpu(),
        "iters": torch.tensor(k + 1),
        "kkt_primal_hist": torch.tensor(kkt_primal_hist),
        "kkt_station_hist": torch.tensor(kkt_station_hist),
        "kkt_compl_hist": torch.tensor(kkt_compl_hist),
    }


# ==========================================================
# 2) Learning with PyTorch: augmented Lagrangian training
#    Recover primal x from learned mu via active-set (periodically + final)
# ==========================================================
def lagrangian(
    lam: torch.Tensor,
    mu: torch.Tensor,
    z: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    r = A.T @ lam - mu
    return 0.5 * torch.dot(lam, lam) + torch.dot(b, lam) + torch.dot(z, r)

@torch.no_grad()
def proj_nonneg_(x: torch.Tensor):
    x.clamp_(min=0.0)


def learning_nnls_dual_by_backprop(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    lr_lam: float = 1e-3,
    lr_mu: float = 1e-3,
    lr_z: float = 1e-4,
    steps: int = 50_000,
    eps: float = 1e-6,
    rho: float = 1.0,

    check_every: int = 200,
    min_improve: float = 1e-3,
    patience: int = 1500,
    lr_decay: float = 0.5,
    min_lr: float = 1e-7,

    # CVXPY reference solution (optimal)
    x_ref: torch.Tensor = None,
    obj_ref: float = None,

    # active-set recovery params
    tau_recover: float = 1e-10,
):
    A, b = A.to(device), b.to(device)
    m, n = A.shape

    # reference solution
    if x_ref is not None:
        x_ref = x_ref.to(device, dtype=A.dtype)


    A_np = A.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()

    lam = torch.nn.Parameter(0.1 * torch.randn(m, dtype=A.dtype, device=device))
    mu  = torch.nn.Parameter(torch.rand(n, dtype=A.dtype, device=device))
    z   = torch.nn.Parameter(torch.zeros(n, dtype=A.dtype, device=device))

    opt_primal = torch.optim.Adam(
        [{"params": [lam], "lr": lr_lam},
         {"params": [mu],  "lr": lr_mu}]
    )
    opt_dual = torch.optim.Adam([z], lr=lr_z)

    best_r = float("inf")
    no_improve_steps = 0

    # Record KKT residuals (using recovered x only at checkpoints; otherwise keep list stable)
    kkt_primal_hist: List[float] = []
    kkt_station_hist: List[float] = []
    kkt_compl_hist: List[float] = []

    # Record gap/dist relative to CVXPY (only at checkpoints)
    obj_hist: List[float] = []
    gap_hist: List[float] = []
    dist_hist: List[float] = []

    # We store one point each checkpoint, so x-axis = checkpoint index
    for k in range(steps):
        # ---- 1) descent on (lam, mu): minimize L_aug ----
        opt_primal.zero_grad()

        r = A.T @ lam - mu
        L = lagrangian(lam, mu, z, A, b)
        L_aug = L + 0.5 * rho * torch.dot(r, r)

        L_aug.backward(retain_graph=True)
        opt_primal.step()
        proj_nonneg_(mu)

        # ---- 2) ascent on z: maximize L ----
        opt_dual.zero_grad()
        L_ascent = lagrangian(lam, mu, z, A, b)
        (-L_ascent).backward()
        opt_dual.step()
        if (k + 1) % 2000 == 0:
            for g in opt_primal.param_groups:
                g["lr"] = max(g["lr"] * lr_decay, min_lr)
            #print("delay",k)
        # ---- 3) stopping on constraint residual ----
        with torch.no_grad():
            r = A.T @ lam - mu
            r_norm = torch.linalg.norm(r).item()


        with torch.no_grad():
            if (k + 1) % check_every == 0:
                # Recover x from current mu (active-set)
                mu_np = mu.detach().cpu().numpy()
                x_np = recover_x_from_mu_active_set(A_np, b_np, mu_np, tau=tau_recover, max_refine=200)
                x_k = torch.from_numpy(x_np).to(device=device, dtype=A.dtype)

                # KKT for primal NNLS
                grad = A.T @ (A @ x_k - b)
                primal_violation = torch.clamp(-x_k, min=0.0)
                dual_violation = torch.clamp(-grad, min=0.0)
                kkt_comp = torch.linalg.norm(x_k * grad).item()

                kkt_primal = torch.linalg.norm(primal_violation).item()
                kkt_stat = torch.linalg.norm(dual_violation).item()

                kkt_primal_hist.append(kkt_primal)
                kkt_station_hist.append(kkt_stat)
                kkt_compl_hist.append(kkt_comp)

                # objective norm
                obj_val = torch.linalg.norm(A @ x_k - b).item()
                obj_hist.append(obj_val)

                if obj_ref is not None:
                    gap_hist.append(abs(obj_val - obj_ref))
                if x_ref is not None:
                    dist_hist.append(torch.linalg.norm(x_k - x_ref).item())

                # logging

                print(
                    f"[BP step {k + 1}] r_norm={r_norm:.3e}, "
                    f"||Ax-b||={obj_val:.3e}, "
                    f"KKT_primal={kkt_primal:.3e}, "
                    f"KKT_stat={kkt_stat:.3e}, "
                    f"KKT_comp={kkt_comp:.3e}"
                )
                if kkt_stat <= eps and kkt_comp <= eps and r_norm <= eps:
                    print(f"[BP] Converged, r_norm={r_norm:.3e}")
                    break

    # final recover x from mu
    with torch.no_grad():
        mu_np = mu.detach().cpu().numpy()
        x_np = recover_x_from_mu_active_set(A_np, b_np, mu_np, tau=tau_recover, max_refine=200)
        x_final = torch.from_numpy(x_np).to(device=device, dtype=A.dtype)

    return (
        lam.detach().cpu(),
        mu.detach().cpu(),
        z.detach().cpu(),
        x_final.detach().cpu(),
        torch.tensor(kkt_primal_hist),
        torch.tensor(kkt_station_hist),
        torch.tensor(kkt_compl_hist),
        torch.tensor(obj_hist),
        torch.tensor(gap_hist),
        torch.tensor(dist_hist),
    )


# ==========================================================
# 3) Main script: multiple (m, n) sizes + plotting
# ==========================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    sizes: List[Tuple[int, int]] = [
        (30, 20),
        (60, 40),
        (90, 60),
        (120, 80),
        (150, 100),
    ]
    lr_rate = [1e-2,1e-2,8e-3,8e-3,5e-3]
    # Record KKT residuals for each method
    admm_kkt_primal_all = {}
    admm_kkt_stat_all   = {}
    admm_kkt_comp_all   = {}
    bp_kkt_primal_all   = {}
    bp_kkt_stat_all     = {}
    bp_kkt_comp_all     = {}

    # Record BP training gaps/dist relative to CVXPY optimal
    bp_gap_all  = {}   # | ||Ax_bp - b|| - ||Ax_cvx - b|| |
    bp_dist_all = {}   # ||x_bp - x_cvx||
    i = 0
    for (m, n) in sizes:
        print(f"\n=== Problem size m={m}, n={n} ===")
        A = torch.randn(m, n, device=device, dtype=torch.float32)
        x_true = torch.rand(n, device=device, dtype=torch.float32)
        x_true[x_true < 0.8] = 0.0
        b = A @ x_true + 0.01 * torch.randn(m, device=device, dtype=torch.float32)

        # ---------- CVXPY optimal reference ----------
        A_cvx = A.detach().cpu().double()
        b_cvx = b.detach().cpu().double()
        x_cvx, obj_cvx,mu_cvx = solve_nnls_cvxpy(A_cvx, b_cvx)
        x_cvx = x_cvx.to(device=device, dtype=A.dtype)
        mu_cvx = mu_cvx.to(device=device, dtype=A.dtype)

        print(f"[CVX  m={m},n={n}] ||Ax-b||={obj_cvx:.6f}")

        # ---------- ADMM (dual) + recover x from mu ----------
        out_admm = nnls_dual_admm(
            A, b, rho=2.0, max_iter=20000,
            eps_abs=1e-6, eps_rel=1e-6, verbose=False,
            tau_recover=1e-6,
        )

        x_admm = out_admm["x"].to(device=device, dtype=A.dtype)
        admm_obj = torch.linalg.norm(A @ x_admm - b).item()
        print(f"[ADMM m={m},n={n}] ||Ax-b||={admm_obj:.6f}, iters={int(out_admm['iters'])}")

        admm_kkt_primal_all[(m, n)] = out_admm["kkt_primal_hist"]
        admm_kkt_stat_all[(m, n)]   = out_admm["kkt_station_hist"]
        admm_kkt_comp_all[(m, n)]   = out_admm["kkt_compl_hist"]

        # ---------- Backprop-based dual learning (recover x from mu active-set) ----------
        (
            lam, mu, z, x_bp,
            kkt_bp_primal, kkt_bp_stat, kkt_bp_comp,
            obj_hist, gap_hist, dist_hist
        ) = learning_nnls_dual_by_backprop(
            A,
            b,
            lr_lam=lr_rate[i],
            lr_mu=lr_rate[i],
            lr_z=lr_rate[i]/10,
            steps=50_000,
            eps=5e-4,
            rho=1,
            check_every=100,
            lr_decay=0.7,
            min_lr=1e-8,
            x_ref=x_cvx,
            obj_ref=obj_cvx,
            tau_recover=1e-10,
        )
        i = i+1
        bp_obj = torch.linalg.norm(A @ x_bp.to(A.dtype).to(device) - b).item()
        print(f"[BP   m={m},n={n}] ||Ax-b||={bp_obj:.6f}, ||x_bp - x_cvx||="
              f"{torch.linalg.norm(x_bp.to(device) - x_cvx).item():.3e} "
              f"||mu_bp - mu_cvx||={torch.linalg.norm(mu.to(device) - mu_cvx).item():.3e}"
              )

        bp_kkt_primal_all[(m, n)] = kkt_bp_primal
        bp_kkt_stat_all[(m, n)]   = kkt_bp_stat
        bp_kkt_comp_all[(m, n)]   = kkt_bp_comp

        bp_gap_all[(m, n)]  = gap_hist
        bp_dist_all[(m, n)] = dist_hist

    # ----------------- BP training gap relative to CVXPY -----------------
    # 1) Objective value distance: | ||Ax_bp - b|| - ||Ax_cvx - b|| |
    plt.figure()
    for (m, n), hist in bp_gap_all.items():
        if len(hist) == 0:
            continue
        iters = np.arange(1, len(hist) + 1) * 100  # because we record every check_every=200
        y = hist.numpy()
        plt.plot(iters, y, label=f"m={m}, n={n}")

    plt.xlabel("Iteration", fontsize=11, fontweight="bold")
    plt.ylabel("| ||Ax_bp - b|| - ||Ax_opt - b|| |", fontsize=11, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=11)
    plt.title("Euclidean Distance to CVXPY Optimal Value During Learning", fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    fig1_name = f"{script_name}_objective_value_distance"
    plt.savefig(os.path.join(fig_dir, fig1_name + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, fig1_name + ".png"), dpi=300, bbox_inches="tight")

    # 2) Solution distance: ||x_bp - x_cvx||
    plt.figure()
    for (m, n), hist in bp_dist_all.items():
        if len(hist) == 0:
            continue
        iters = np.arange(1, len(hist) + 1) * 200
        y = hist.numpy()
        plt.plot(iters, y, label=f"m={m}, n={n}")

    plt.xlabel("Iteration", fontsize=11, fontweight="bold")
    plt.ylabel("||x_bp - x_opt||_2", fontsize=11, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=11)
    plt.title("Euclidean Distance to CVXPY Optimal Solution During Learning", fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    fig2_name = f"{script_name}_solution_distance"
    plt.savefig(os.path.join(fig_dir, fig2_name + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, fig2_name + ".png"), dpi=300, bbox_inches="tight")

    # 3) Objective value distance (log y-scale)
    EPS_LOG = 1e-16

    plt.figure()
    for (m, n), hist in bp_gap_all.items():
        if len(hist) == 0:
            continue
        iters = np.arange(1, len(hist) + 1) * 200
        y = hist.numpy() + EPS_LOG  # avoid log(0)
        plt.plot(iters, y, label=f"m={m}, n={n}")

    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=11, fontweight="bold")
    plt.ylabel("| ||Ax_bp - b|| - ||Ax_opt - b|| | (log scale)",
               fontsize=11, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=11)
    plt.title("Euclidean Distance to CVXPY Optimal Value (Log Scale)",
              fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)

    fig3_name = f"{script_name}_objective_value_distance_log"
    plt.savefig(os.path.join(fig_dir, fig3_name + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, fig3_name + ".png"), dpi=300, bbox_inches="tight")

    # 4) Solution distance (log y-scale)
    plt.figure()
    for (m, n), hist in bp_dist_all.items():
        if len(hist) == 0:
            continue
        iters = np.arange(1, len(hist) + 1) * 200
        y = hist.numpy() + EPS_LOG  # avoid log(0)
        plt.plot(iters, y, label=f"m={m}, n={n}")

    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=11, fontweight="bold")
    plt.ylabel("||x_bp - x_opt||_2 (log scale)",
               fontsize=11, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=11)
    plt.title("Euclidean Distance to CVXPY Optimal Solution (Log Scale)",
              fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)

    fig4_name = f"{script_name}_solution_distance_log"
    plt.savefig(os.path.join(fig_dir, fig4_name + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, fig4_name + ".png"), dpi=300, bbox_inches="tight")

    plt.show()
