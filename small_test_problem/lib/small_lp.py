from small_test_problem.lib.functions import *


def solve_lp(s_init, time_horizon):
    m = gp.Model()
    m.Params.LogToConsole = 0

    state_idx = [(j, w, t) for j in J for w in waiting_times for t in range(time_horizon)]
    state_j_idx = [(j, t) for j in J for t in range(time_horizon)]
    x = m.addVars(state_idx, name="x")
    x_j = m.addVars(state_j_idx, name="x_j")
    s = m.addVars(state_idx, name="s")

    # Initial conditions
    m.addConstrs(s[j, w, 0] == s_init[j, w] for j in J for w in waiting_times)

    # Constraints
    # Positivity
    m.addConstrs(x[j, w, t] >= 0 for j in J for w in waiting_times for t in range(time_horizon))
    # x_j definition
    m.addConstrs(x_j[j, t] == sum(x[j, w, t] for w in waiting_times) for j in J for t in range(time_horizon))
    # Queue capacity constraint
    m.addConstrs(x[j, w, t] <= s[j, w, t] for j in J for w in waiting_times for t in range(time_horizon))
    # Resource capacity constraint
    m.addConstrs(sum(x_j[j, t] for j in r_queues[r]) <= eta[r] for t in range(time_horizon) for r in R)
    # Untreated patients wait
    m.addConstrs(s[j, 2, t] == sum(s[j, w, t - 1] - x[j, w, t - 1] for w in waiting_times[1:])
                 for j in J for t in range(1, time_horizon))
    m.addConstrs(s[j, 1, t] == s[j, 0, t-1] - x[j, 0, t-1] for j in J for t in range(1, time_horizon))
    # New patients every week
    m.addConstrs(s[0, 0, t] == num_new_patients for t in range(1, time_horizon))
    # Transition patients
    m.addConstrs(s[j, 0, t] == sum(q_full[(i, j)] * x_j[i, t-1] for i in J)
                 for j in J[1:] for t in range(1, time_horizon))

    # Objective function
    total_cost = sum(cost(j, U[j], w) * (s[j, w, t] - x[j, w, t])
                     for j in J for w in waiting_times for t in range(time_horizon))
    total_reward = sum(reward[j] * x_j[j, t] for j in J for t in range(time_horizon))

    m.setObjective(total_reward - total_cost, GRB.MAXIMIZE)
    m.optimize()

    first_action = [int(x[j, w, 0].X) for j in J for w in range(W)]
    return first_action
