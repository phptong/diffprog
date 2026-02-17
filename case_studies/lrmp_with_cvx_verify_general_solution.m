clear; clc; close all;
rng(0);

%% Problem setup
m = 50;
n = 30;
A = randn(m,n);
b = rand(m,1) + 0.5;

%% Build weighted graph Laplacian (connected graph)
W = zeros(n);

for i = 1:n-1
    W(i,i+1) = 0.5 + rand();
    W(i+1,i) = W(i,i+1);
end

extra_edges = round(n);
for k = 1:extra_edges
    i = randi(n); j = randi(n);
    if i ~= j
        W(i,j) = W(i,j) + 0.2*rand();
        W(j,i) = W(i,j);
    end
end

W(1:n+1:end) = 0;
d = sum(W,2);
L = diag(d) - W;
L = (L + L')/2;

fprintf('=== Laplacian Properties ===\n');
fprintf('Symmetry ||L-L''||_F = %.3e\n', norm(L-L','fro'));
e = eig(full(L));
fprintf('Min eigenvalue  = %.3e (should be ~0)\n', min(e));
fprintf('Rank(L)         = %d (should be n-1=%d for connected graph)\n', rank(full(L)), n-1);
fprintf('L*1             = %.3e (should be ~0)\n\n', norm(L*ones(n,1)));

%% CVX solve LR-NNLS
cvx_begin quiet
    variables x(n) y(m)
    dual variables lambda mu
    minimize( 0.5*sum_square(y) + 0.5*quad_form(x, L) )
    subject to
        lambda : y == A*x - b;
        mu     : x >= 0;
cvx_end

fprintf('=== CVX Solution ===\n');
fprintf('Status: %s\n', cvx_status);
fprintf('Optimal value: %.6e\n\n', cvx_optval);

%% Verify KKT conditions
fprintf('=== KKT Conditions Verification ===\n');

% Stationarity: Lx* + A^T*lambda* - mu* = 0
kkt_stationarity = norm(L*x + A'*lambda - mu);
fprintf('Stationarity ||Lx* + A^T lambda* - mu*||  = %.3e\n', kkt_stationarity);

% Complementary slackness: mu^T * x = 0
comp_slack = abs(mu' * x);
fprintf('Complementarity |mu*^T x*|                = %.3e\n', comp_slack);

% Primal feasibility
fprintf('Nonnegativity min(x*)                     = %.3e\n', min(x));
fprintf('Constraint ||Ax* - b - y*||               = %.3e\n\n', norm(A*x - b - y));

%% Compute the pseudoinverse reconstruction
Ldag = pinv(L);
v = mu - A'*lambda;        % Right-hand side: mu* - A^T lambda*
z = Ldag * v;              % Particular solution: z = L^dagger (mu* - A^T lambda*)

%% Method: Least squares fit for c
% Find c to minimize ||x - z - c*1||^2; Solution: c = mean(x - z)
fprintf('=== Method: Least Squares Fit (c) ===\n');

c_method2 = mean(x - z);
fprintf('c (least squares) = %.6e\n', c_method2);

x_method2 = z + c_method2 * ones(n,1);
error_m2 = norm(x - x_method2);
rel_error_m2 = error_m2 / max(norm(x), 1e-16);

fprintf('Reconstruction error    = %.3e\n', error_m2);
fprintf('Relative error          = %.3e\n\n', rel_error_m2);

%% Baseline: Without c (to show necessity)
fprintf('=== Baseline: Without c (to show necessity) ===\n');

x_baseline = z;  % No c*1 term
error_baseline = norm(x - x_baseline);
rel_error_baseline = error_baseline / max(norm(x), 1e-16);

fprintf('Reconstruction error ||x* - z||  = %.3e\n', error_baseline);
fprintf('Relative error                   = %.3e\n', rel_error_baseline);

amp = error_baseline / max(error_m2, 1e-16);
fprintf('Error amplification factor (baseline / LS) = %.1f\n\n', amp);

%% Summary table (ASCII-safe version)
fprintf('================================================================================\n');
fprintf('                         RECONSTRUCTION SUMMARY                                 \n');
fprintf('================================================================================\n');
fprintf(' Method                  | Abs Error    | Rel Error    | Min Value         \n');
fprintf('--------------------------------------------------------------------------------\n');
fprintf(' Least Squares (c)       | %.4e | %.4e | %+.4e      \n', error_m2, rel_error_m2, min(x_method2));
fprintf(' Without c (Baseline)    | %.4e | %.4e | %+.4e      \n', error_baseline, rel_error_baseline, min(x_baseline));
fprintf('================================================================================\n\n');

%% Final verification checklist (console only)
fprintf('================================================================================\n');
fprintf('                     THEORETICAL VALIDATION CHECKLIST                           \n');
fprintf('================================================================================\n');
fprintf(' 1. KKT stationarity satisfied?       %s (residual = %.3e)\n', ...
    iff(kkt_stationarity < 1e-5, '[YES]', '[NO] '), kkt_stationarity);
fprintf(' 2. Complementary slackness holds?    %s (|mu^T x| = %.3e)\n', ...
    iff(comp_slack < 1e-5, '[YES]', '[NO] '), comp_slack);
fprintf(' 3. c is necessary (baseline worse)?  %s (baseline %.1fx worse)\n', ...
    iff(error_baseline > 10*error_m2, '[YES]', '[NO] '), amp);
fprintf('================================================================================\n');

function s = iff(cond, true_str, false_str)
    if cond
        s = true_str;
    else
        s = false_str;
    end
end
