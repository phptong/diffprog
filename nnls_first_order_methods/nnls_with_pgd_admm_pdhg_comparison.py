import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cvxpy as cp

# ============================================================
# Utility Functions
# ============================================================

def power_iteration_ATA(A, num_iters=50):
    m, n = A.shape
    x = np.random.randn(n)
    x /= np.linalg.norm(x) + 1e-12
    for _ in range(num_iters):
        x = A.T @ (A @ x)
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            break
        x /= norm
    return float(x @ (A.T @ (A @ x)))


def objective(A, b, x):
    r = A @ x - b
    return 0.5 * float(r @ r)


def grad_f(A, b, x):
    return A.T @ (A @ x - b)


# ---------------- KKT Residual for NNLS ----------------

def kkt_residual_nnls(A, b, x):
    """
    Primal-only KKT residual using mu = max(0, grad) as the closest feasible dual.
    """
    g = grad_f(A, b, x)
    mu = np.maximum(0.0, g)
    r_stationarity = np.linalg.norm(g - mu)
    r_complementarity = np.linalg.norm(x * mu)
    r_primal = np.linalg.norm(np.minimum(x, 0.0))
    return np.sqrt(r_stationarity**2 + r_complementarity**2 + r_primal**2)


def kkt_residual_with_mu(A, b, x, mu):
    """
    Full KKT residual when mu is provided.
    Stationarity: grad - mu = 0, primal/dual feasibility, complementarity.
    """
    g = grad_f(A, b, x)
    r_stationarity = np.linalg.norm(g - mu)
    r_complementarity = np.linalg.norm(x * mu)
    r_primal = np.linalg.norm(np.minimum(x, 0.0))
    r_dual = np.linalg.norm(np.minimum(mu, 0.0))
    return np.sqrt(r_stationarity**2 + r_complementarity**2 + r_primal**2 + r_dual**2)


# ============================================================
# CVXPY Reference Solver (robust dual via KKT reconstruction)
# ============================================================

def solve_nnls_cvxpy(A, b, solver_preference=("OSQP", "ECOS", "SCS")):
    """
    Solve NNLS via CVXPY to get x_star.
    For dual, use KKT-consistent reconstruction:
        mu_star := max(0, grad_f(x_star))
    This avoids solver-dependent dual sign conventions.
    """
    m, n = A.shape
    x = cp.Variable(n, nonneg=True)
    obj = 0.5 * cp.sum_squares(A @ x - b)
    prob = cp.Problem(cp.Minimize(obj))

    last_err = None
    for s in solver_preference:
        try:
            prob.solve(solver=s, verbose=False)
            if x.value is not None:
                break
        except Exception as e:
            last_err = e
            continue

    if x.value is None:
        raise RuntimeError(f"CVXPY failed to solve NNLS. Last error: {last_err}")

    x_star = np.asarray(x.value).reshape(-1)

    # ---- Robust "gold" dual: project gradient to nonnegative orthant ----
    g = grad_f(A, b, x_star)
    mu_star = np.maximum(0.0, g)

    diag = {
        "status": prob.status,
        "obj": float(prob.value),
        "grad_norm": float(np.linalg.norm(g)),
        "mu_star_min": float(np.min(mu_star)),
        "mu_star_max": float(np.max(mu_star)),
        "mu_star_norm": float(np.linalg.norm(mu_star)),
        "kkt_with_mu": float(kkt_residual_with_mu(A, b, x_star, mu_star)),
        "kkt_primal_only": float(kkt_residual_nnls(A, b, x_star)),
    }
    return x_star, mu_star, diag


# ============================================================
# Method 1: PDG for NNLS KKT system
# ============================================================

def solve_pdg(A, b, x_star, mu_star, max_iter=500, tau=None, sigma=None):
    """
    PDG update:
        grad_x = grad_f(x) - mu
        x <- P_{>=0}(x - tau*grad_x)
        mu <- P_{>=0}(mu - sigma*x)   (dual ascent for constraint -x <= 0)

    For dual tracking & comparison, use the KKT-consistent reconstruction
        mu_k := max(0, grad_f(x_k))
    to avoid ambiguity of PDG's internal mu in degenerate cases.
    """
    m, n = A.shape
    L = power_iteration_ATA(A)
    if tau is None:
        tau = 1.0 / (L + 1e-8)
    if sigma is None:
        sigma = 1.0 / (L + 1e-8)

    x = np.zeros(n)
    mu = np.zeros(n)

    obj_hist = []
    dist_hist = []
    kkt_hist = []
    dual_err_hist = []

    for _ in range(max_iter):
        gx = grad_f(A, b, x)
        grad_x = gx - mu
        x = np.maximum(0.0, x - tau * grad_x)
        mu = np.maximum(0.0, mu - sigma * x)

        # Dual used for comparison/visualization (KKT-reconstructed)
        mu_k = np.maximum(0.0, grad_f(A, b, x))
        dual_err_hist.append(np.linalg.norm(mu_k - mu_star))

        obj_hist.append(objective(A, b, x))
        dist_hist.append(np.linalg.norm(x - x_star))
        kkt_hist.append(kkt_residual_nnls(A, b, x))

    return {
        "x": x,
        "mu": mu,
        "obj": np.array(obj_hist),
        "dist": np.array(dist_hist),
        "kkt": np.array(kkt_hist),
        "dual_err": np.array(dual_err_hist),
    }


# ============================================================
# Method 2: ADMM in (x, y, lam) form
# ============================================================

def solve_admm(A, b, x_star, mu_star, max_iter=500, rho=1.0, inner_x_steps=3):
    """
    ADMM for:
        min 0.5||y||^2
        s.t. Ax - b = y, x >= 0

    Dual for NNLS KKT comparison:
        mu_k := max(0, A^T lam_k) (with sign alignment handled outside if needed)
    """
    m, n = A.shape
    L = power_iteration_ATA(A)
    tau_x = 1.0 / (L + 1e-8)

    x = np.zeros(n)
    y = np.zeros(m)
    lam = np.zeros(m)

    obj_hist = []
    dist_hist = []
    kkt_hist = []
    dual_err_hist = []

    for _ in range(max_iter):
        # x-update (inexact)
        b_tilde = b + y - lam / rho
        for _ in range(inner_x_steps):
            gx = A.T @ (A @ x - b_tilde)
            x = np.maximum(0.0, x - tau_x * gx)

        # y-update
        Ax_minus_b = A @ x - b
        y = (lam + rho * Ax_minus_b) / (1.0 + rho)

        # lam-update
        r = A @ x - b - y
        lam = lam + rho * r

        # Dual used for comparison/visualization (projected)
        mu_k = A.T @ lam
        mu_k = np.maximum(mu_k, 0.0)
        dual_err_hist.append(np.linalg.norm(mu_k - mu_star))

        obj_hist.append(objective(A, b, x))
        dist_hist.append(np.linalg.norm(x - x_star))
        kkt_hist.append(kkt_residual_nnls(A, b, x))

    return {
        "x": x,
        "y": y,
        "lam": lam,
        "obj": np.array(obj_hist),
        "dist": np.array(dist_hist),
        "kkt": np.array(kkt_hist),
        "dual_err": np.array(dual_err_hist),
    }


# ============================================================
# Method 3: PDHG for Fenchel-type saddle formulation
# ============================================================

def solve_pdhg(A, b, x_star, mu_star, max_iter=500, tau=None, sigma=None):
    """
    PDHG:
        x^{k+1} = P_{>=0}(x^k - tau A^T y^k)
        x̄^{k+1} = 2x^{k+1} - x^k
        y^{k+1} = ( y^k + sigma(A x̄^{k+1} - b) ) / (1+sigma)

    Dual for NNLS KKT comparison:
        mu_k := max(0, A^T y_k)
    """
    m, n = A.shape
    L = power_iteration_ATA(A)

    if tau is None or sigma is None:
        base = 0.9 / (np.sqrt(L) + 1e-8)
        tau = base
        sigma = base

    x = np.zeros(n)
    y = np.zeros(m)

    obj_hist = []
    dist_hist = []
    kkt_hist = []
    dual_err_hist = []

    for _ in range(max_iter):
        x_new = np.maximum(0.0, x - tau * (A.T @ y))
        x_bar = 2.0 * x_new - x
        y = (y + sigma * (A @ x_bar - b)) / (1.0 + sigma)
        x = x_new

        mu_k = A.T @ y
        mu_k = np.maximum(mu_k, 0.0)
        dual_err_hist.append(np.linalg.norm(mu_k - mu_star))

        obj_hist.append(objective(A, b, x))
        dist_hist.append(np.linalg.norm(x - x_star))
        kkt_hist.append(kkt_residual_nnls(A, b, x))

    return {
        "x": x,
        "y": y,
        "obj": np.array(obj_hist),
        "dist": np.array(dist_hist),
        "kkt": np.array(kkt_hist),
        "dual_err": np.array(dual_err_hist),
    }


# ============================================================
# Output Directory Helper
# ============================================================

def get_output_dir():
    try:
        script_path = Path(__file__).resolve()
        out_dir = script_path.parent / script_path.stem
    except NameError:
        out_dir = Path.cwd() / "nnls_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(fig, out_dir, basename):
    png_path = out_dir / f"{basename}.png"
    pdf_path = out_dir / f"{basename}.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# Dual comparison helpers
# ============================================================

def compare_duals(tag, mu_hat, mu_star, atol=1e-6, rtol=1e-4):
    abs_e = np.linalg.norm(mu_hat - mu_star)
    denom = np.linalg.norm(mu_star) + 1e-12
    rel_e = abs_e / denom
    ok = (abs_e <= atol) or (rel_e <= rtol)

    print(f"[Dual Check] {tag}")
    print(f"  ||mu_hat - mu_star||_2 = {abs_e:.3e}")
    print(f"  ||mu_star||_2          = {np.linalg.norm(mu_star):.3e}")
    print(f"  rel_err               = {rel_e:.3e}")
    print(f"  PASS?                 = {ok}")
    return ok, abs_e, rel_e


# ============================================================
# Main Experiment Script
# ============================================================

def run_experiment(m=150, n=100, sparsity=0.2,
                   noise_level=0.01, max_iter=200, seed=0):

    np.random.seed(seed)
    out_dir = get_output_dir()

    # Generate random NNLS test instance
    A = np.random.randn(m, n) / np.sqrt(m)
    x_true = np.random.rand(n)
    mask = np.random.rand(n) < sparsity
    x_true = x_true * mask

    b_clean = A @ x_true
    noise = (noise_level * np.linalg.norm(b_clean)
             * np.random.randn(m) / (np.sqrt(m) + 1e-12))
    b = b_clean + noise

    # ---------------- CVXPY reference ----------------
    x_star, mu_star, cvxpy_diag = solve_nnls_cvxpy(A, b)
    print("\n=== CVXPY Reference (x_star + mu_star via KKT reconstruction) ===")
    print(f"Status: {cvxpy_diag['status']}")
    print(f"Objective (CVXPY): {cvxpy_diag['obj']:.6e}")
    print(f"KKT residual (with mu_star): {cvxpy_diag['kkt_with_mu']:.3e}")
    print(f"KKT residual (primal-only): {cvxpy_diag['kkt_primal_only']:.3e}")
    print(f"mu_star range: [{cvxpy_diag['mu_star_min']:.3e}, {cvxpy_diag['mu_star_max']:.3e}]")
    print(f"||mu_star||_2: {cvxpy_diag['mu_star_norm']:.3e}")
    print(f"||grad||_2   : {cvxpy_diag['grad_norm']:.3e}")

    # Run methods
    res_pdg  = solve_pdg(A, b, x_star, mu_star, max_iter=max_iter)
    res_admm = solve_admm(A, b, x_star, mu_star, max_iter=max_iter)
    res_pdhg = solve_pdhg(A, b, x_star, mu_star, max_iter=max_iter)

    # Build final comparable mu's for console checks
    mu_pdg = np.maximum(0.0, grad_f(A, b, res_pdg["x"]))      # PDG: reconstructed
    mu_admm = A.T @ res_admm["lam"]
    mu_pdhg = A.T @ res_pdhg["y"]

    # Optional sign alignment for ADMM/PDHG
    g_star = grad_f(A, b, x_star)
    if np.linalg.norm(g_star - mu_admm) > np.linalg.norm(g_star + mu_admm):
        mu_admm = -mu_admm
    if np.linalg.norm(g_star - mu_pdhg) > np.linalg.norm(g_star + mu_pdhg):
        mu_pdhg = -mu_pdhg

    mu_admm = np.maximum(mu_admm, 0.0)
    mu_pdhg = np.maximum(mu_pdhg, 0.0)

    print("\n=== Final objective values ===")
    print(f"  CVXPY : {objective(A, b, x_star):.6e}")
    print(f"  PDG   : {res_pdg['obj'][-1]:.6e}")
    print(f"  ADMM  : {res_admm['obj'][-1]:.6e}")
    print(f"  PDHG  : {res_pdhg['obj'][-1]:.6e}")

    print("\n=== Final distance to x_star (CVXPY) ===")
    print(f"  PDG  : {res_pdg['dist'][-1]:.6e}")
    print(f"  ADMM : {res_admm['dist'][-1]:.6e}")
    print(f"  PDHG : {res_pdhg['dist'][-1]:.6e}")

    print("\n=== Final KKT residual (primal-only metric) ===")
    print(f"  PDG  : {res_pdg['kkt'][-1]:.6e}")
    print(f"  ADMM : {res_admm['kkt'][-1]:.6e}")
    print(f"  PDHG : {res_pdhg['kkt'][-1]:.6e}")

    print("\n=== Dual consistency checks vs mu_star (KKT-reconstructed) ===")
    compare_duals("PDG  (mu=proj grad)", mu_pdg,  mu_star, atol=1e-5, rtol=1e-3)
    compare_duals("ADMM (mu=A^T lam)",   mu_admm, mu_star, atol=1e-5, rtol=1e-3)
    compare_duals("PDHG (mu=A^T y)",     mu_pdhg, mu_star, atol=1e-5, rtol=1e-3)

    print("\n=== Strong KKT residual using each method's mu ===")
    print(f"  PDG  KKT(x,mu): {kkt_residual_with_mu(A, b, res_pdg['x'],  mu_pdg):.3e}")
    print(f"  ADMM KKT(x,mu): {kkt_residual_with_mu(A, b, res_admm['x'], mu_admm):.3e}")
    print(f"  PDHG KKT(x,mu): {kkt_residual_with_mu(A, b, res_pdhg['x'], mu_pdhg):.3e}")

    iters = np.arange(1, max_iter + 1)

    # ---------------- Figure 1: Objective vs Iteration ----------------
    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.semilogy(iters, res_pdg["obj"],  label="PDG")
    ax1.semilogy(iters, res_admm["obj"], label="ADMM")
    ax1.semilogy(iters, res_pdhg["obj"], label="PDHG")
    ax1.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"$0.5\|Ax-b\|_2^2$", fontsize=11, fontweight="bold")
    ax1.set_title("Objective Value vs Iteration", fontsize=11, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=11)
    ax1.legend()
    ax1.grid(True, which='major', ls='--', alpha=0.5)
    ax1.grid(False, which='minor')
    fig1.tight_layout()
    save_figure(fig1, out_dir, "nnls_pdg_admm_pdhg_objective_value")

    # ---------------- Figure 2: Euclidean Distance to x_star (CVXPY) ----------------
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.semilogy(iters, res_pdg["dist"],  label="PDG")
    ax2.semilogy(iters, res_admm["dist"], label="ADMM")
    ax2.semilogy(iters, res_pdhg["dist"], label="PDHG")
    ax2.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"$\|x_k - x_\star\|_2$", fontsize=11, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=11)
    ax2.set_title("Euclidean Distance to Primal Solution (CVXPY)", fontsize=11, fontweight="bold")
    ax2.legend()
    ax2.grid(True, which='major', ls='--', alpha=0.5)
    ax2.grid(False, which='minor')
    fig2.tight_layout()
    save_figure(fig2, out_dir, "nnls_pdg_admm_pdhg_distance_to_xstar")

    # ---------------- Figure 3: KKT Residual vs Iteration ----------------
    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.semilogy(iters, res_pdg["kkt"],  label="PDG")
    ax3.semilogy(iters, res_admm["kkt"], label="ADMM")
    ax3.semilogy(iters, res_pdhg["kkt"], label="PDHG")
    ax3.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax3.set_ylabel("KKT Residual", fontsize=11, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=11)
    ax3.set_title("KKT Residual vs Iteration", fontsize=11, fontweight="bold")
    ax3.legend()
    ax3.grid(True, which='major', ls='--', alpha=0.5)
    ax3.grid(False, which='minor')
    fig3.tight_layout()
    save_figure(fig3, out_dir, "nnls_pdg_admm_pdhg_kkt_residual")

    # ---------------- Figure 4: Euclidean Distance to mu_star (CVXPY) ----------------
    fig4 = plt.figure(figsize=(6, 4))
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.semilogy(iters, res_pdg["dual_err"],  label="PDG")
    ax4.semilogy(iters, res_admm["dual_err"], label="ADMM")
    ax4.semilogy(iters, res_pdhg["dual_err"], label="PDHG")
    ax4.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax4.set_ylabel(r"$\|\mu_k - \mu_\star\|_2$", fontsize=11, fontweight="bold")
    ax4.set_title("Euclidean Distance to Dual Solution (CVXPY)", fontsize=11, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=11)
    ax4.legend()
    ax4.grid(True, which='major', ls='--', alpha=0.5)
    ax4.grid(False, which='minor')
    fig4.tight_layout()
    save_figure(fig4, out_dir, "nnls_pdg_admm_pdhg_distance_to_mustar")

    plt.show()


if __name__ == "__main__":
    run_experiment(m=150, n=100, max_iter=200, seed=0)
