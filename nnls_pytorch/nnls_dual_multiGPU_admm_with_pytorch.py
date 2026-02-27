import os
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.distributed as dist
import cvxpy as cp


# ==========================================================
# 0) Correct recovery (active-set from mu)
# ==========================================================
def recover_x_from_mu_active_set(
    A: np.ndarray,
    b: np.ndarray,
    mu_init: np.ndarray,
    tau: float = 1e-6,
    max_refine: int = 50,
) -> np.ndarray:
    m, n = A.shape
    mu = mu_init.reshape(-1).copy()
    assert mu.shape[0] == n

    F = (mu <= tau)
    x = np.zeros(n)

    for _ in range(max_refine):
        F_old = F.copy()

        idx = np.where(F)[0]
        x[:] = 0.0
        if idx.size > 0:
            x_F, *_ = np.linalg.lstsq(A[:, idx], b, rcond=None)
            x[idx] = x_F

        # If any negative in free set, drop them (classic cleanup)
        neg = np.where(x < 0)[0]
        if neg.size > 0:
            F[neg] = False
            x[neg] = 0.0

        # Recompute KKT multipliers from current x
        g = A.T @ (A @ x - b)
        mu = np.maximum(0.0, g)

        # Update free set using refreshed mu
        F = (mu <= tau)
        F[x > 0] = True

        if np.array_equal(F, F_old):
            break

    x = np.maximum(x, 0.0)
    return x


# ==========================================================
# 1) CVXPY baseline (rank0 only)
# ==========================================================
def solve_nnls_cvxpy(A: np.ndarray, b: np.ndarray,
                    solver_preference=("OSQP", "ECOS", "SCS")) -> Dict[str, np.ndarray]:
    """
    Solve NNLS primal:
        min_x 0.5||Ax-b||^2  s.t. x>=0
    Return x_star and useful duals (KKT-reconstructed).
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

    if x.value is None:
        raise RuntimeError(f"CVXPY failed. Last error: {last_err}")

    x_star = np.asarray(x.value).reshape(-1)
    r = A @ x_star - b
    lam_residual = r                                # residual-type dual
    mu_kkt = np.maximum(0.0, A.T @ r)               # multiplier for x>=0 (solver-agnostic)

    return {
        "x_star": x_star,
        "obj": float(prob.value),
        "status": prob.status,
        "lam_residual": lam_residual,
        "mu_kkt": mu_kkt,
    }


# ==========================================================
# 2) Distributed utilities
# ==========================================================
def dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if dist_is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if dist_is_initialized() else 1

def barrier():
    if dist_is_initialized():
        dist.barrier()

def allreduce_sum_(x: torch.Tensor):
    if dist_is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)

def setup_distributed_from_env():
    """
    Works with:
      torchrun --nproc_per_node=NUM_GPUS script.py
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    return 0, 1, 0


# ==========================================================
# 3) Multi-GPU (distributed) batched ADMM
#    Partition blocks across GPUs, global consensus via all_reduce
# ==========================================================
def nnls_dual_admm_batch_distributed(
    A_local: torch.Tensor,   # (N_local, m, n) on this rank
    B_local: torch.Tensor,   # (N_local, m)    on this rank
    N_total: int,
    rho: float = 1.0,
    rho_c: float = 1.0,
    max_iter: int = 2000,
    eps_abs: float = 1e-7,
    eps_rel: float = 1e-7,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Distributed consensus ADMM (block-splitting).
    Each rank holds a subset of blocks. We compute global Z by averaging via all_reduce.
    """

    assert A_local.ndim == 3 and B_local.ndim == 2
    device = A_local.device
    dtype = A_local.dtype

    N_local, m, n = A_local.shape
    A_T = A_local.transpose(1, 2)            # (N_local, n, m)
    AAT = torch.bmm(A_local, A_T)            # (N_local, m, m)

    # ADMM variables on each rank
    Lambda = torch.zeros((N_local, m), dtype=dtype, device=device)
    Z_i    = torch.zeros((N_local, n), dtype=dtype, device=device)
    U      = torch.zeros((N_local, n), dtype=dtype, device=device)
    V      = torch.zeros((N_local, n), dtype=dtype, device=device)

    # Global consensus variable (replicated on each rank)
    Z = torch.zeros(n, dtype=dtype, device=device)

    # Pre-factorize (I + rho A A^T) for each local block
    I_m = torch.eye(m, dtype=dtype, device=device).unsqueeze(0)  # (1,m,m)
    M = I_m + rho * AAT                                          # (N_local,m,m)
    L = torch.linalg.cholesky(M)

    coeff_rho = rho / (rho + rho_c)
    coeff_rc  = rho_c / (rho + rho_c)

    # constants for stopping
    sqrt_pri = torch.sqrt(torch.tensor(2.0 * N_total * n, dtype=dtype, device=device))
    sqrt_dual = sqrt_pri

    for k in range(max_iter):
        Z_i_old = Z_i.clone()


        rhs = -B_local + rho * torch.bmm(A_local, (Z_i - U).unsqueeze(-1)).squeeze(-1)
        Lambda_new = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)

        A_T_lambda = torch.bmm(A_T, Lambda_new.unsqueeze(-1)).squeeze(-1)

        Z_i = coeff_rho * (A_T_lambda + U) + coeff_rc * (Z - V)

        sum_local = (Z_i + V).sum(dim=0)  # (n,)
        allreduce_sum_(sum_local)
        Z = torch.clamp(sum_local / float(N_total), min=0.0)

        U_new = U + (A_T_lambda - Z_i)
        V_new = V + (Z_i - Z)

        r1 = A_T_lambda - Z_i
        r2 = Z_i - Z

        r_sq_local = torch.linalg.norm(r1, "fro") ** 2 + torch.linalg.norm(r2, "fro") ** 2
        r_sq = r_sq_local.clone()
        allreduce_sum_(r_sq)
        r_norm = torch.sqrt(r_sq)

        # dual residual (approx)
        dZ = (Z_i - Z_i_old)
        s_local = rho * torch.linalg.norm(torch.bmm(A_local, dZ.unsqueeze(-1)).squeeze(-1), "fro")
        s_sq = (s_local ** 2).clone()
        allreduce_sum_(s_sq)
        s_norm = torch.sqrt(s_sq)

        # thresholds
        max_term_pri_local = (
            torch.linalg.norm(A_T_lambda, "fro") ** 2 +
            torch.linalg.norm(Z_i, "fro") ** 2 +
            torch.linalg.norm(Z, 2) ** 2
        )
        max_term_pri = max_term_pri_local.clone()
        allreduce_sum_(max_term_pri)
        max_term_pri = torch.sqrt(max_term_pri)

        e_pri = sqrt_pri * eps_abs + eps_rel * max_term_pri

        max_term_dual_local = (
            (rho * torch.linalg.norm(U_new, "fro")) ** 2 +
            (rho_c * torch.linalg.norm(V_new, "fro")) ** 2
        )
        max_term_dual = max_term_dual_local.clone()
        allreduce_sum_(max_term_dual)
        max_term_dual = torch.sqrt(max_term_dual)

        e_dual = sqrt_dual * eps_abs + eps_rel * max_term_dual

        # update
        Lambda, U, V = Lambda_new, U_new, V_new

        if verbose and get_rank() == 0 and (k % 100 == 0 or k == max_iter - 1):
            print(f"iter {k+1:4d} | r={r_norm.item():.3e}, s={s_norm.item():.3e}, "
                  f"e_pri={e_pri.item():.3e}, e_dual={e_dual.item():.3e}")

        if (r_norm <= e_pri) and (s_norm <= e_dual):
            if verbose and get_rank() == 0:
                print(f"Converged at iteration {k+1}.")
            break

    return {
        "Lambda_local": Lambda,  # local only
        "Z": Z,                  # global consensus on each rank
        "iters": torch.tensor(k + 1, device=device),
    }


# ==========================================================
# 4) Demo / test
# ==========================================================
def main():
    rank, world_size, local_rank = setup_distributed_from_env()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"world_size={world_size}, device={device}")

    torch.manual_seed(0)
    np.random.seed(0)

    # Global problem
    M, n = 320, 80
    N = world_size
    assert M % N == 0, "M must be divisible by N"
    m = M // N

    # Generate on rank0 and broadcast to all ranks
    if rank == 0:
        A_full = torch.randn(M, n, dtype=torch.float64)
        x_true = torch.rand(n, dtype=torch.float64)
        x_true[x_true < 0.8] = 0.0
        b_full = A_full @ x_true + 0.01 * torch.randn(M, dtype=torch.float64)
    else:
        A_full = torch.empty(M, n, dtype=torch.float64)
        b_full = torch.empty(M, dtype=torch.float64)

    if dist_is_initialized():
        dist.broadcast(A_full, src=0)
        dist.broadcast(b_full, src=0)

    # Partition blocks: rank r gets rows [r*m:(r+1)*m]
    A_block = A_full[rank * m:(rank + 1) * m, :].contiguous().to(device)
    b_block = b_full[rank * m:(rank + 1) * m].contiguous().to(device)

    A_local = A_block.unsqueeze(0)  # (1,m,n)
    B_local = b_block.unsqueeze(0)  # (1,m)

    # Run distributed ADMM
    out = nnls_dual_admm_batch_distributed(
        A_local, B_local,
        N_total=N,
        rho=1.0,
        rho_c=1.0,
        max_iter=10000,
        eps_abs=1e-8,
        eps_rel=1e-8,
        verbose=True,
    )

    Z = out["Z"]  # on each rank


    if rank == 0:
        A_np = A_full.cpu().numpy()
        b_np = b_full.cpu().numpy()

        cvx = solve_nnls_cvxpy(A_np, b_np)
        x_cvx = cvx["x_star"]
        mu_cvx = cvx['mu_kkt']

        # Build mu from Z (KKT multiplier proxy), then recover x by your method
        Z_np = Z.detach().cpu().numpy()
        mu_from_Z = np.maximum(0.0, A_np.T @ (A_np @ Z_np - b_np))
        x_rec = recover_x_from_mu_active_set(A_np, b_np, mu_from_Z, tau=1e-5, max_refine=2000)

        # Compare
        def obj_half(x):
            r = A_np @ x - b_np
            return 0.5 * float(r @ r)

        print("\n[CVXPY Baseline]")
        print(f"  status = {cvx['status']}")
        print(f"  obj(x_cvx) = {obj_half(x_cvx):.6e}")

        print("\n[ADMM Consensus Z]")
        # print(f"  obj(Z)      = {obj_half(Z_np):.6e}")
        print(f"  ||Z - x_cvx||_2 = {np.linalg.norm(Z_np - mu_cvx):.6e}")

        print("\n[Recovered x via mu(active-set)]")
        print(f"  obj(x_rec)  = {obj_half(x_rec):.6e}")
        print(f"  ||x_rec - x_cvx||_2 = {np.linalg.norm(x_rec - x_cvx):.6e}")

    barrier()
    if dist_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
