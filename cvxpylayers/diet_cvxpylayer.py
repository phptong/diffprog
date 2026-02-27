import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

torch.manual_seed(0)

n = 5
m = 3

A = (torch.rand(m, n) * 10.0).double().requires_grad_()
b = (torch.rand(m) * 20.0 + 5.0).double().requires_grad_()
c = (torch.rand(n) * 2.0 + 0.1).double().requires_grad_()

# ===== CVXPY layer =====
x = cp.Variable(n)

A_param = cp.Parameter((m, n))
b_param = cp.Parameter(m)
c_param = cp.Parameter(n)

eps_reg = 1e-6
objective = cp.Minimize(c_param @ x + 0.5 * eps_reg * cp.sum_squares(x))
constraints = [A_param @ x >= b_param, x >= 0]
problem = cp.Problem(objective, constraints)

layer = CvxpyLayer(problem, parameters=[A_param, b_param, c_param], variables=[x])

# forward
x_star, = layer(A, b, c, solver_args={
    "max_iters": 200000,
    "eps": 1e-9,
    "verbose": True
})
print("Optimal x* =", x_star)

# loss + backward
loss = (x_star**2).sum()
loss.backward()

print("\nGradients:")
print("grad A norm:", A.grad.norm().item())
print("grad b norm:", b.grad.norm().item())
print("grad c norm:", c.grad.norm().item())