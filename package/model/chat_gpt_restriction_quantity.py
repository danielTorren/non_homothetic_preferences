import numpy as np
from scipy.optimize import minimize

# Parameters
alpha_m = np.array([0.6, 0.5, 0.7])  # Share parameters for low-carbon goods in each sector
rho = 0.5  # Substitution parameter between L_m and H_m
theta = 0.8  # Substitution parameter across sectors
w_m = np.array([0.3, 0.4, 0.3])  # Weight for each sector m
p_L_m = np.array([1.0, 1.2, 1.1])  # Prices of low-carbon goods in each sector
p_H_m = np.array([1.5, 1.8, 1.6])  # Prices of high-carbon goods in each sector
X_m = np.array([5.0, 4.0, 6.0])  # Minimum consumption levels in each sector
B = 100  # Total budget
M = len(w_m)  # Number of sectors

# CES Utility Function for each sector with sector-specific alpha_m
def ces_sector(L_m, H_m, alpha_m, rho):
    return (alpha_m * L_m**rho + (1 - alpha_m) * H_m**rho)**(1/rho)

# Total Utility Function (Nested CES)
def utility(x, alpha_m, rho, theta, w_m):
    L_m = x[:M]
    H_m = x[M:]
    sector_utilities = [w_m[m] * ces_sector(L_m[m], H_m[m], alpha_m[m], rho)**(theta) for m in range(M)]
    return -(sum(sector_utilities))**(1/theta)  # Negative for maximization problem

# Constraints
def budget_constraint(x, p_L_m, p_H_m, B):
    L_m = x[:M]
    H_m = x[M:]
    return B - np.sum(p_L_m * L_m + p_H_m * H_m)  # Should be zero for equality constraint

def min_consumption_constraint(x, X_m):
    L_m = x[:M]
    H_m = x[M:]
    return L_m + H_m - X_m  # Should be >= 0 for inequality constraint

# Initial guess for L_m and H_m
x0 = np.concatenate([np.ones(M) * 5, np.ones(M) * 5])  # Initial guess of L_m and H_m

# Bounds for L_m and H_m (non-negative values)
bounds = [(0, None)] * (2 * M)

# Define the constraints
constraints = [
    {'type': 'eq', 'fun': budget_constraint, 'args': (p_L_m, p_H_m, B)},
    {'type': 'ineq', 'fun': min_consumption_constraint, 'args': (X_m,)}
]

# Solve the optimization problem
result = minimize(
    utility,
    x0,
    args=(alpha_m, rho, theta, w_m),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Results
if result.success:
    L_opt = result.x[:M]
    H_opt = result.x[M:]
    print("Optimal quantities of low-carbon goods (L_m):", L_opt)
    print("Optimal quantities of high-carbon goods (H_m):", H_opt)
    print("Maximized utility:", -result.fun)
else:
    print("Optimization failed:", result.message)
