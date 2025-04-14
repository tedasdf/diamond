import numpy as np
from scipy.optimize import minimize

# Define state and action spaces
states = np.arange(5)  # [0, 1, 2, 3, 4]
actions = np.array([-1, 0, 1])  # left, stay, right
action_indices = {-1: 0, 0: 1, 1: 2}
num_states = len(states)
num_actions = len(actions)

# Example cost functions (adjust as needed)
c_x = np.array([1, 0.5, 0.25, 0.5, 0])  # State cost
def c_u(u): return 0.1 * u**2  # Action cost

# Example trained model (replace with actual model)
def trained_model(x, u):
    p = np.ones(num_states) / num_states  # Uniform for simplicity
    return p

# Hyperparameters
c = 1.0  # Prior strength
eta_0 = 0.1  # Base ambiguity radius
M = 10  # Planning horizon
N = 100  # Maximum steps

# Initialize memory
counts = np.zeros((num_states, num_actions, num_states))
p_bar_initial = np.zeros((num_states, num_actions, num_states))
for x in states:
    for u_idx, u in enumerate(actions):
        p_bar_initial[x, u_idx, :] = trained_model(x, u)
alpha = c * p_bar_initial

# Functions to compute p_bar_k and eta_k
def get_p_bar_k(x, u):
    u_idx = action_indices[u]
    total = np.sum(alpha[x, u_idx, :] + counts[x, u_idx, :])
    return (alpha[x, u_idx, :] + counts[x, u_idx, :]) / total

def get_eta_k(x, u):
    u_idx = action_indices[u]
    total_counts = np.sum(counts[x, u_idx, :])
    return eta_0 / (1 + total_counts)

# Updated compute_ambiguity_cost
def compute_ambiguity_cost(x_k_minus_1, u_k, p_bar_k_func, eta_k_func, f=None):
    p_bar_k = p_bar_k_func(x_k_minus_1, u_k)
    eta = eta_k_func(x_k_minus_1, u_k)
    if f is None:
        f = c_x

    support = np.where(p_bar_k > 0)[0]
    if len(support) == 1:
        return f[support[0]]

    p_bar_support = p_bar_k[support]
    f_support = f[support]

    def objective(p):
        return -np.dot(p, f_support)

    def kl_constraint(p):
        eps = 1e-10
        p_safe = np.clip(p, eps, 1)
        p_bar_safe = np.clip(p_bar_support, eps, 1)
        kl = np.sum(p_safe * np.log(p_safe / p_bar_safe))
        return eta - kl

    constraints = [
        {'type': 'eq', 'fun': lambda p: np.sum(p) - 1},
        {'type': 'ineq', 'fun': kl_constraint}
    ]
    bounds = [(0, 1)] * len(support)
    p0 = p_bar_support

    result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return -result.fun
    else:
        print(f"Optimization failed at x={x_k_minus_1}, u={u_k}")
        return np.inf

# Backward pass for policy computation
def backward_pass(N_plan, p_bar_k_func, eta_k_func):
    c_hat = np.zeros((N_plan + 1, num_states))
    policy = np.zeros((N_plan, num_states), dtype=int)
    c_hat[N_plan, :] = c_x  # Terminal cost

    for k in range(N_plan - 1, -1, -1):
        for x in states:
            min_cost = float('inf')
            best_u = 0
            for u in actions:
                f = c_x + c_hat[k + 1, :]
                cost = c_u(u) + compute_ambiguity_cost(x, u, p_bar_k_func, eta_k_func, f)
                if cost < min_cost:
                    min_cost = cost
                    best_u = u
            c_hat[k, x] = min_cost
            policy[k, x] = best_u
    return c_hat, policy

# Simulation with memory updates
x = 0  # Starting state
trajectory = [x]
for k in range(N):
    if x == 4:  # Goal state
        break
    N_plan = min(M, N - k)
    _, policy = backward_pass(N_plan, get_p_bar_k, get_eta_k)
    u_k = policy[0, x]
    p_true = trained_model(x, u_k)  # Simulate true dynamics
    x_next = np.random.choice(states, p=p_true)
    counts[x, action_indices[u_k], x_next] += 1
    x = x_next
    trajectory.append(x)

print("Trajectory:", trajectory)    