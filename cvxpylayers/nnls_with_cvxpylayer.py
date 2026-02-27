import numpy as np
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# -----------------------------
# NNLS via CVXPYLayers (double)
#   minimize_{x>=0} 0.5 ||A x - b||_2^2
# -----------------------------
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=6, suppress=True)

# sizes
m, n = 30, 15

# -----------------------------
# 1) Build CVXPY problem
# -----------------------------
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)

objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b))
constraints = [x >= 0]
problem = cp.Problem(objective, constraints)

print("is_dpp =", problem.is_dpp())  # should be True for NNLS

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

# solver settings (tight for good feasibility)
solver_args = {
    "eps": 1e-10,
    "max_iters": 200000,
    "acceleration_lookback": 0
}

# -----------------------------
# 2) Forward solve once
# -----------------------------
A_t = torch.randn(m, n, requires_grad=True)
b_t = torch.randn(m, requires_grad=True)

(x_star,) = layer(A_t, b_t, solver_args=solver_args)

print("\n=== Forward (CVXPYLayers) ===")
print("x_star dtype:", x_star.dtype)
print("min(x*) =", float(x_star.min()))
print("neg_part_norm =", float(torch.norm(torch.clamp(-x_star, min=0.0))))
print("||Ax-b|| =", float(torch.norm(A_t @ x_star - b_t)))

# optional: clamp for reporting only (do NOT use for training loss)
x_out = torch.clamp(x_star.detach(), min=0.0)
print("min(clamped x*) =", float(x_out.min()))

# -----------------------------
# 3) Simple outer loss + backward
#    loss = 0.5 ||x*||^2
# -----------------------------
loss = 0.5 * torch.sum(x_star**2)
loss.backward()

print("\n=== Backprop demo ===")
print("loss =", float(loss.detach()))
print("||dLoss/dA||_F =", float(torch.norm(A_t.grad)))
print("||dLoss/db||_2 =", float(torch.norm(b_t.grad)))
print("dLoss/db (first 5) =", b_t.grad[:5].detach().cpu().numpy())