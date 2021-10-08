from small_test_problem.lib.patient import *
import gurobipy as gp
from gurobipy import GRB


def phi(s):
    return np.array([[i for i in s]]).T


def cost(j, u, w):
    if w < u:
        return 0
    else:
        return wait_cost[j] * w / (u + 1)


def contribution(s, a):
    r = sum(sum(a[(i * W) + w] for w in range(W)) * reward[i] for i in J)
    c = sum(cost(j, U[j], w) * (s[(j * W) + w] - a[(j * W) + w]) for j in J for w in waiting_times)
    return r - c


def contribution_evi(s_before_wait, a):
    r = sum(a[i] * reward[i] for i in J)
    c = sum(cost(j, U[j], w) * s_before_wait[(j * 3) + w] for j in J for w in waiting_times)
    return r - c


def pi(s, theta, gamma):
    model = gp.Model()
    model.Params.LogToConsole = 0
    idx = list(range(len(J) * W))
    a = model.addVars(idx, name="action", vtype=GRB.INTEGER)
    exp_state = model.addVars(idx, name="expected state")
    F = len(theta)
    # Constraints
    model.addConstrs(exp_state[(j * W) + w] == s[(j * W) + w-1] - a[(j * W) + w-1] for j in J for w in range(1, W))
    model.addConstrs(exp_state[j * W] == sum(q_full[i, j] * sum(a[(i * W) + w] for w in range(W)) for i in J)
                     for j in [1, 2])
    model.addConstr(exp_state[0] == num_new_patients + sum(q_full[i, 0] * sum(a[(i * W) + w] for w in range(W))
                                                           for i in J))
    model.addConstrs(0 <= a[i] for i in idx)
    model.addConstrs(a[i] <= s[i] for i in idx)
    model.addConstrs(sum(a[i] for i in range(W * r_queues[r][0], W * (r_queues[r][-1] + 1))) <= eta[r] for r in R)

    # Objective
    dot = sum(phi(exp_state)[i][0] * theta[i][0] for i in range(F))
    model.setObjective(contribution(s, a) + gamma * dot, GRB.MAXIMIZE)
    model.optimize()
    return [int(a[i].X) for i in idx]


def generate_dataset(M, ub_patients):
    D = []
    for i in range(M):
        num_patients = np.random.randint(ub_patients)
        patient_list = [generate_random_patient(new=False, short=True) for _ in range(num_patients)]
        patients = {(j, w): [] for j in J for w in waiting_times}
        s = np.zeros(len(J) * W)
        for p in patient_list:
            j = p.current
            w = p.waiting_time
            patients[j, w] += [p]
            s[(j * W) + w] += 1
        D += [(s, patients)]
    return D


def pre_transition(s, a):
    # Provides the state before patient transitions occur
    # Remove treated patients from queues in order of highest waiting time
    for j in J:
        leftover = a[j]
        for w in [2, 1, 0]:
            s_jw = s[(j * W) + w]
            treated = min(s_jw, leftover)
            leftover -= treated
            s[(j * W) + w] = s_jw - treated
    s_before_wait = s.copy()
    # Untreated patients wait one time unit
    for j in J:
        for w in [2, 1]:
            s[(j * W) + w] = min(s[(j * W) + w] + s[(j * W) + w - 1], l[j])
            s[(j * W) + w - 1] = 0
    # Add new patients from outside
    s[0] = num_new_patients
    return s, s_before_wait


def transition(s_pre, x):
    # Add patients to pre-transition state according to transition x
    for (i, j) in x.keys():
        if j < len(J):
            s_pre[j * W] = min(s_pre[j * W] + x[(i, j)], l[j])
    return s_pre


def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation


def possible_actions(s):
    actions = []
    for [a, b, c] in all_actions:
        if (a <= sum(s[w] for w in range(W))) and (b <= sum(s[W + w] for w in range(W))) and \
                (c <= sum(s[(2 * W) + w] for w in range(W))):
            actions += [[a, b, c]]
    return actions


def possible_transitions(a):
    X_0 = list(sums(eta['OD'], a[0]))
    X_1 = list(sums(eta['OR'], a[1]))
    X_2 = list(sums(eta['OR'], a[2]))
    X = [{(0, 0): a[0], (0, 1): a[1], (0, 2): a[2], (0, 3): a[3],
          (1, 0): b[0], (1, 3): b[1],
          (2, 0): c[0], (2, 3): c[1]}
         for a in X_0 for b in X_1 for c in X_2]
    return X


def norm(v):
    return np.sqrt(sum(i[1] ** 2 for i in v.items()))
