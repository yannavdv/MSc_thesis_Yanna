from lib.functions import *


def solve_lp(s_init, time_horizon, gamma=1, alpha=0):
    m = gp.Model()
    m.Params.LogToConsole = 0

    state_idx = [(j, u, w, t) for j in J for u in U[j] for w in range(W[j, u]) for t in range(time_horizon)]
    state_j_idx = [(j, u, t) for j in J for u in U[j] for t in range(time_horizon)]
    x = m.addVars(state_idx, name="x")
    x_j = m.addVars(state_j_idx, name="x_j")
    s = m.addVars(state_idx, name="s")
    y = m.addVars([(j, t) for j in r_queues['OD'] for t in range(time_horizon)], name="y", vtype=GRB.BINARY)

    # Initial conditions
    m.addConstrs(s[(*i, 0)] == s_init[i] for i in idx)

    # Constraints
    # Positivity
    m.addConstrs(x[(*i, t)] >= 0 for i in idx for t in range(time_horizon))
    # x_j definition
    m.addConstrs(x_j[j, u, t] == sum(x[j, u, w, t] for w in range(W[j, u]))
                 for j in J for u in U[j] for t in range(time_horizon))
    # Queue capacity constraint
    m.addConstrs(x[(*i, t)] <= s[(*i, t)] for i in idx for t in range(time_horizon))
    # Resource capacity constraint
    m.addConstrs(sum(zeta[j, r] * sum(x[j, u, w, t] for u in U[j] for w in range(W[j, u])) for j in J) <= eta[r]
                 for r in R for t in range(time_horizon))
    # Untreated patients wait
    m.addConstrs(s[j, u, W[j, u]-1, t] == sum(s[j, u, w, t - 1] - x[j, u, w, t - 1] for w in [W[j, u] - 2, W[j, u] - 1])
                 for j in J for u in U[j] for t in range(1, time_horizon))
    m.addConstrs(s[j, u, w, t] == s[j, u, w-1, t-1] - x[j, u, w-1, t-1]
                 for j in J for u in U[j] for w in range(1, W[j, u]-1) for t in range(1, time_horizon))
    # Transition patients & add new patients
    m.addConstrs(s[j, u, 0, t] == (sum(q[(i, v), (j, u)] * x_j[i, v, t-1] for i in J for v in U[i]) +
                                   (lambdas[(j, u)] * num_new_patients)) for j in J for u in U[j]
                 for t in range(1, time_horizon))

    if alpha > 0:  # Hybrid method
        minimal = {'FC': alpha * 30, 'RC': alpha * 52, 'DC': alpha * 9}
        K = 1000
        m.addConstrs(minimal[j] - sum(s[j, u, w, t] for u in U[j] for w in range(W[j, u])) <= K*y[j, t]
                     for j in r_queues['OD'] for t in range(time_horizon))
        m.addConstrs(sum(s[j, u, w, t] for u in U[j] for w in range(W[j, u])) - minimal[j] <= K*(1-y[j, t])
                     for j in r_queues['OD'] for t in range(time_horizon))
        m.addConstrs(sum(x_j[j, u, t] for u in U[j]) >=
                     sum(s[j, u, w, t] for u in U[j] for w in range(W[j, u])) - K*(1-y[j, t])
                     for j in r_queues['OD'] for t in range(time_horizon))
        m.addConstrs(sum(x_j[j, u, t] for u in U[j]) >= minimal[j] - K*y[j, t]
                     for j in r_queues['OD'] for t in range(time_horizon))

    # Objective function
    total_cost = sum(sum(cost(j, u, w) * (s[j, u, w, t] - x[j, u, w, t])
                     for j in J for u in U[j] for w in range(W[j, u])) * (gamma ** t) for t in range(time_horizon))
    total_reward = sum(sum(reward[j] * x_j[j, u, t]
                           for j in J for u in U[j]) * (gamma ** t) for t in range(time_horizon))

    m.setObjective(total_reward - total_cost, GRB.MAXIMIZE)
    m.optimize()

    return {i: x[(*i, 0)].X for i in idx}
