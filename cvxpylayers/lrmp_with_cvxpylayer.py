"""
---Differentiable LR-NNLS via CVXPYLayers
This script constructs and differentiates the following optimization problem:
    minimize_{x >= 0}
        1/2 ||A x - b||_2^2
      + 1/2 x^T L x
      + ε/2 ||x||_2^2
Decision variable:
    x ∈ R^n  (nonnegative)
Parameters:
    A ∈ R^{m×n}  (data matrix, differentiable)
    b ∈ R^m      (observation vector, differentiable)
    L ∈ R^{n×n}  (fixed graph Laplacian, PSD)
    ε > 0        (small ridge term ensuring strong convexity)
Purpose:
    1) Solve LR-NNLS using CVXPYLayers.
    2) Compare with CVXPY (OSQP) reference for numerical accuracy.
    3) Verify differentiability by defining an outer loss:
            loss = 1/2 ||x*||_2^2
       and backpropagating gradients through the optimization layer
       to A and b via implicit differentiation of KKT conditions.
The ε term guarantees uniqueness and stable gradient computation.
"""

import numpy as np
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

np.set_printoptions(precision=12, suppress=True)

# -----------------------------
# example fixed L
# -----------------------------
def build_connected_laplacian(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(n - 1):
        w = 0.5 + rng.random()
        W[i, i + 1] = w
        W[i + 1, i] = w
    for _ in range(int(round(n))):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            W[i, j] += 0.2 * rng.random()
            W[j, i] = W[i, j]
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    L = np.diag(d) - W
    return 0.5 * (L + L.T)

def obj_value_np(A, b, L, eps, x):
    r = A @ x - b
    return 0.5 * (r @ r) + 0.5 * (x @ (L @ x)) + 0.5 * eps * (x @ x)

def rel_err(a, b, eps=1e-12):
    na = np.linalg.norm(a)
    return np.linalg.norm(a - b) / max(na, eps)

# -----------------------------
# sizes / data
# -----------------------------
m, n = 50, 30
rng = np.random.default_rng(0)
L_np = build_connected_laplacian(n, seed=0)

# use double everywhere
dtype = torch.float64

A_t = torch.randn(m, n, dtype=dtype)
b_t = torch.rand(m, dtype=dtype) + 0.5
eps_t = torch.tensor(1e-3, dtype=dtype)  # your choice

A_np = A_t.detach().cpu().numpy()
b_np = b_t.detach().cpu().numpy()
eps_np = float(eps_t.item())

# -----------------------------
# CVXPYLayers model (L is Constant -> DPP)
# -----------------------------
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)

L_const = cp.Constant(L_np)
eps = cp.Parameter(nonneg=True)

objective = cp.Minimize(
    0.5 * cp.sum_squares(A @ x - b)
    + 0.5 * cp.quad_form(x, L_const)
    + 0.5 * eps * cp.sum_squares(x)
)
constraints = [x >= 0]
problem = cp.Problem(objective, constraints)
print("is_dpp =", problem.is_dpp())

layer = CvxpyLayer(problem, parameters=[A, b, eps], variables=[x])

# -----------------------------
# Solve via CVXPYLayers
# -----------------------------
solver_args = {
    "eps": 1e-10,
    "max_iters": 200000,
    "acceleration_lookback": 0
}

(x_layer_raw,) = layer(A_t, b_t, eps_t, solver_args=solver_args)
x_layer_raw_np = x_layer_raw.detach().cpu().numpy()
min_raw = x_layer_raw_np.min()
neg_norm_raw = np.linalg.norm(np.minimum(x_layer_raw_np, 0.0))

# clamp for strict nonnegativity (post-processing)
x_layer_clamp = torch.clamp(x_layer_raw, min=0.0)
x_layer_clamp_np = x_layer_clamp.detach().cpu().numpy()
min_clamp = x_layer_clamp_np.min()
neg_norm_clamp = np.linalg.norm(np.minimum(x_layer_clamp_np, 0.0))

obj_layer_raw = obj_value_np(A_np, b_np, L_np, eps_np, x_layer_raw_np)
obj_layer_clamp = obj_value_np(A_np, b_np, L_np, eps_np, x_layer_clamp_np)

print("\n=== CVXPYLayers ===")
print("dtype:", x_layer_raw.dtype)
print(f"min(x_raw)        = {min_raw:.3e}")
print(f"neg_part_norm(raw)= {neg_norm_raw:.3e}")
print(f"objective(raw)    = {obj_layer_raw:.12e}")
print(f"min(x_clamp)      = {min_clamp:.3e}")
print(f"neg_part_norm(cl) = {neg_norm_clamp:.3e}")
print(f"objective(clamp)  = {obj_layer_clamp:.12e}")

# -----------------------------
# Reference solve via CVXPY + OSQP
# -----------------------------
x_ref = cp.Variable(n)
prob_ref = cp.Problem(
    cp.Minimize(
        0.5 * cp.sum_squares(A_np @ x_ref - b_np)
        + 0.5 * cp.quad_form(x_ref, L_np)
        + 0.5 * eps_np * cp.sum_squares(x_ref)
    ),
    [x_ref >= 0]
)

# OSQP tolerances: tighten a bit so feasibility is very clean
prob_ref.solve(
    solver=cp.OSQP,
    verbose=False,
    eps_abs=1e-10,
    eps_rel=1e-10,
    max_iter=200000
)

if prob_ref.status not in ["optimal", "optimal_inaccurate"]:
    raise RuntimeError(f"CVXPY(OSQP) failed: status={prob_ref.status}")

x_cvx = x_ref.value.reshape(-1)
obj_cvx = obj_value_np(A_np, b_np, L_np, eps_np, x_cvx)

print("\n=== CVXPY (OSQP) reference ===")
print("status:", prob_ref.status)
print(f"min(x_cvx)        = {x_cvx.min():.3e}")
print(f"neg_part_norm(cvx)= {np.linalg.norm(np.minimum(x_cvx, 0.0)):.3e}")
print(f"objective(cvx)    = {obj_cvx:.12e}")

# quick KKT sanity check for CVXPY solution:
# stationarity: (A^T(Ax-b) + Lx + eps x) - mu = 0, with mu>=0 and mu_i x_i = 0
# mu_cvx = prob_ref.constraints[0].dual_value.reshape(-1)  # dual for x>=0
# grad = A_np.T @ (A_np @ x_cvx - b_np) + L_np @ x_cvx + eps_np * x_cvx
# station_res = np.linalg.norm(grad - mu_cvx)
# comp = abs(mu_cvx @ x_cvx)
# print("\n--- CVXPY KKT sanity ---")
# print(f"stationarity ||grad - mu|| = {station_res:.3e}")
# print(f"complementarity |mu^T x|   = {comp:.3e}")
# print(f"min(mu)                    = {mu_cvx.min():.3e}")

# -----------------------------
# Compare CVXPYLayers vs CVXPY
# -----------------------------
re_raw = rel_err(x_cvx, x_layer_raw_np)
re_cl = rel_err(x_cvx, x_layer_clamp_np)

print("\n=== Comparison (Layer vs CVX) ===")
print(f"rel_err(raw)   = {re_raw:.3e}")
print(f"rel_err(clamp) = {re_cl:.3e}")
print(f"obj_gap(raw)   = {obj_layer_raw - obj_cvx:.3e}")
print(f"obj_gap(clamp) = {obj_layer_clamp - obj_cvx:.3e}")

# -----------------------------
# Final verdict
# -----------------------------
tol_feas = 1e-8
tol_match = 1e-6
tol_obj = 1e-6

feas_ok = neg_norm_raw < tol_feas
match_ok = re_raw < tol_match
obj_ok = abs(obj_layer_raw - obj_cvx) < tol_obj

print("\n" + "="*70)
print("                         FINAL VERDICT")
print("="*70)
print(f"Feasibility (||neg|| < {tol_feas:.0e})      : {'[PASS]' if feas_ok else '[FAIL]'}")
print(f"Match CVX (rel_err < {tol_match:.0e})      : {'[PASS]' if match_ok else '[FAIL]'}")
print(f"Objective gap < {tol_obj:.0e}              : {'[PASS]' if obj_ok else '[FAIL]'}")
print("="*70)

if feas_ok and match_ok and obj_ok:
    print("Overall conclusion: CVXPYLayers solution matches CVXPY(OSQP) to high precision.")
else:
    print("Overall conclusion: Minor discrepancies detected (check tolerances or solver settings).")


# ============================================
# Simple differentiable demo: loss = 0.5 ||x||^2
# ============================================
print("\n=== Backprop demo: loss = 0.5 ||x||^2 ===")

dtype = torch.float64

# Make A and b learnable
A_t = torch.randn(m, n, dtype=dtype, requires_grad=True)
b_t = (torch.rand(m, dtype=dtype) + 0.5).requires_grad_(True)
eps_t = torch.tensor(1e-3, dtype=dtype)

solver_args = {
    "eps": 1e-10,
    "max_iters": 200000,
    "acceleration_lookback": 0
}

# Forward through optimization layer
(x_star,) = layer(A_t, b_t, eps_t, solver_args=solver_args)

# Simple smooth loss
loss = 0.5 * torch.sum(x_star ** 2)

# Backprop
loss.backward()

print("loss =", float(loss.detach()))
print("||dLoss/dA||_F =", float(torch.norm(A_t.grad)))
print("||dLoss/db||_2 =", float(torch.norm(b_t.grad)))

# Show first few gradient entries
print("dLoss/db (first 5) =", b_t.grad[:5].detach().cpu().numpy())