# Learning to Optimize by Differentiable Programming

This repository contains the source code for first-order methods, PyTorch implementations, and representative case studies developed for the tutorial "Learning to Optimize by Differentiable Programming".
The tutorial focuses on integrating differentiable programming with optimization, combining classical first-order methods such as ADMM and PDHG with Fenchel–Rockafellar and Lagrangian duality, and implementing these ideas in modern frameworks including PyTorch, TensorFlow, and JAX. The accompanying case studies illustrate these concepts in practice.
For the content details, please refer to our paper "Learning to Optimize by Differentiable Programming" (https://arxiv.org/abs/2601.16510).


### Installation
The following software and libraries are required:
- Python 3.11  
- PyTorch 2.5.1  
- CUDA 12.1 (torch.version.cuda = 12.1)  
- MATLAB R2018a  
- CVX (for MATLAB)
- CVXPY (for Python)


### Repository Structure
├── nnls_first_order_methods/
│   └── nnls_with_pgd_admm_pdhg_comparison.py
│       (Compares PGD, PDHG, and ADMM for solving the NNLS problem.)
│
├── nnls_pytorch/
│   ├── nnls_dual_admm_with_pytorch.py
│   │   (PyTorch implementation of the dual-form ADMM solver for NNLS.)
│   ├── nnls_dual_loss_learning_refer_to_cvxpy_with_pytorch.py
│   │   (Learning framework for NNLS using CVXPY solutions as references.)
│   └── nnls_dual_multiGPU_admm_with_pytorch.py
│       (Multi-GPU ADMM implementation for large-scale NNLS.)
│
├── case_studies/
│   ├── diet_dladmm_learning_with_cvxpy.py
│   ├── diet_dladmm_with_cvxpy.py
│   ├── diet_pdhg_with_cvxpy.py
│   │   (Three methods for solving the Stigler Diet problem.)
│   ├── nnv_with_cvxpy.py
│   │   (Solving the neural network verification problem.)
│   ├── opf_with_cvxpy.py
│   │   (Solving the optimal power flow problem.)
│   └── lrmp_with_cvx_verify_general_solution.m
│       (MATLAB script verifying the dual-driven primal solution reconstruction for LR-NNLS.)
│
└── cvxpylayers/
    ├── nnls_with_cvxpylayer.py
    │   (Differentiable NNLS using CVXPYLayers.)
    ├── diet_cvxpylayer.py
    │   (Linear program solved via CVXPYLayers with gradient backpropagation.)
    └── lrmp_with_cvxpylayer.py
        (Differentiable LR-NNLS with Laplacian regularization using CVXPYLayers.)

## Usage

### Running the Python script

The script `nnls_with_pgd_admm_pdhg_comparison.py` can be executed in two ways.  
It relies on `cvxpy` to solve the reference convex optimization problems, and can be used with standard solvers such as ECOS, SCS, or MOSEK.
1. Direct Execution  
   Open the file in a Python IDE (e.g., PyCharm) and run it directly.
2. Command-Line Execution  
   From a terminal, navigate to the script directory and run:
   ```bash
   python nnls_with_pgd_admm_pdhg_comparison.py

### Running the MATLAB script
The script `lrmp_with_cvx_verify_general_solution.m` is used to verify the general solution of the LRMP and can be executed as follows. It requires MATLAB with the CVX toolbox installed and properly configured.
1. Direct Execution  
   Open the file `lrmp_with_cvx_verify_general_solution.m` in MATLAB and run it directly.
2. Command-Line Execution  
   From the MATLAB command window or a system terminal, navigate to the script directory and run:
   
   ```matlab
   lrmp_with_cvx_verify_general_solution

### Citing

@article{tao2026learning,
  title={Learning to Optimize by Differentiable Programming},
  author={Tao, Liping and Tong, Xindi and Tan, Chee Wei},
  journal={arXiv preprint arXiv:2601.16510},
  year={2026}
}