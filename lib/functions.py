from lib.patient import *
from lib.parameters.params_smk import *
import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB


def cost(j, u, w):
    if w < u:
        return 0
    else:
        return omega[j] * w / u


def contribution(s, a):
    c = sum(cost(j, u, w) * (s[(j, u, w)] - a[(j, u, w)]) for (j, u, w) in idx)
    r = sum(reward[j] * sum(a[(j, u, w)] for u in U[j] for w in range(W[j, u])) for j in J)
    return r - c


def phi(s):
    return np.array([[s[i] for i in idx]]).T


def pi(s, theta, gamma):
    model = gp.Model()
    model.Params.LogToConsole = 0
    a = model.addVars(idx, name="action", vtype=GRB.INTEGER)
    exp_state = model.addVars(idx, name="expected state")
    F = len(theta)

    # Constraints
    model.addConstrs(exp_state[j, u, 0] == sum(q[(i, v), (j, u)] * a[i, v, w] for i in J for v in U[i]
                                               for w in range(W[i, v])) + (lambdas[j, u] * num_new_patients)
                     for j in J for u in U[j])
    model.addConstrs(exp_state[j, u, w] == s[j, u, w-1] - a[j, u, w-1]
                     for j in J for u in U[j] for w in range(1, W[j, u] - 1))
    model.addConstrs(exp_state[j, u, W[j, u] - 1] == sum(s[j, u, w] - a[j, u, w] for w in [W[j, u] - 2, W[j, u] - 1])
                     for j in J for u in U[j])
    model.addConstrs(0 <= a[i] for i in idx)
    model.addConstrs(a[i] <= s[i] for i in idx)
    model.addConstrs(sum(zeta[j, r] * sum(a[(j, u, w)] for u in U[j] for w in range(W[j, u])) for j in J) <= eta[r]
                     for r in R)

    # Objective
    dot = sum(phi(exp_state)[i][0] * theta[i][0] for i in range(F))
    model.setObjective(contribution(s, a) + gamma * dot, GRB.MAXIMIZE)
    model.optimize()

    return {i: int(a[i].X) for i in idx}


def highest_contribution(state):
    action = {i: 0 for i in idx}
    for r in R:
        r_patients = [(j, u, w) for j in r_queues[r] for u in U[j]
                      for w in range(W[j, u]) for _ in range(int(state[j, u, w]))]
        r_patients.sort(key=lambda x: (reward[x[0]] + cost(*x)) / zeta[x[0], r], reverse=True)
        capacity = eta[r]
        for p in r_patients:
            capacity -= zeta[p[0], r]
            if capacity < 0:
                break
            action[p] += 1
    return action


def generate_patient(new, short):
    path = random.choice(all_paths)
    if new:
        stage = 0
        waiting_time = 0
    else:
        stage = random.choice(range(len(path)))
        waiting_time = min(int(np.random.exponential(path[stage][1])), W[path[stage][0], path[stage][1]] - 1)
    if short:
        return ShortPatient(path, stage, waiting_time)
    else:
        return Patient(path, stage, waiting_time)


def generate_dataset(M, mu, sigma):
    D = []
    for i in range(M):
        num_patients = int(np.random.normal(mu, sigma))
        patient_list = [generate_patient(new=False, short=True) for _ in range(num_patients)]
        patients = {i: [] for i in idx}
        s = {i: 0 for i in idx}
        for p in patient_list:
            patients[p.current] += [p]
            s[p.current] += 1
        D += [(s, patients)]
    return D


def predict_state(current_state, compact_actions, start_time, end_time):
    pred_states = {(*i, t): 0 for i in idx for t in range(start_time, end_time + 1)}
    for i in idx:
        pred_states[(*i, start_time)] = current_state[i]
    for t in range(start_time, end_time):
        for j in J:
            sorted_patients = [(u, w) for u in U[j] for w in range(W[j, u])
                               for _ in range(int(pred_states[j, u, w, t]))]
            sorted_patients.sort(key=lambda x: cost(j, *x))
            total = len(sorted_patients)
            a = min(compact_actions[j, t], total)
            for _ in range(a):
                (u, w) = sorted_patients.pop()
                for i in J:
                    for v in U[i]:
                        pred_states[i, v, 0, t + 1] += q[(j, u), (i, v)]
            for _ in range(a, total):
                (u, w) = sorted_patients.pop()
                pred_states[j, u, min(w + 1, W[j, u] - 1), t + 1] += 1
            for u in U[j]:
                # Expected new patients
                pred_states[j, u, 0, t + 1] += lambdas[(j, u)] * num_new_patients
    return {i: pred_states[(*i, end_time)] for i in idx}


def init_state(initial_patients, time=None, new_patients=None):
    if time is None:
        states = {i: 0 for i in idx}
    else:
        states = {(*i, t): 0 for i in idx for t in range(time + 1)}
    patients = {i: [] for i in idx}

    # Generate patients for initial state
    for _ in range(initial_patients):
        p = generate_patient(new=False, short=False)
        if time is None:
            states[p.get_stage()] += 1
        else:
            states[(*p.get_stage(), 0)] += 1
        patients[p.get_stage()].append(p)

    if new_patients is None:
        return states, patients
    else:
        return states, patients, [generate_patient(new=True, short=False) for _ in range(time * new_patients)]


def perform_compact_action(states, patients, t, action, redistribute=False):
    true_action = {i: 0 for i in idx}
    c = 0
    sorted_patients = {}
    for j in J:
        sorted_patients[j] = [p for u in U[j] for w in range(W[j, u]) for p in patients[j, u, w]]
        sorted_patients[j].sort(key=lambda x: cost(*x.get_stage()))
    for j in J:
        total = len(sorted_patients[j])
        a = min(total, action[j])
        c += a * reward[j]
        for _ in range(a):  # Treat patients
            p = sorted_patients[j].pop()
            patients[p.get_stage()].remove(p)
            true_action[p.get_stage()] += 1
            j = p.get_stage()[0]
            p.treat()
            if p.active:
                patients[p.get_stage()].insert(0, p)
                states[(*p.get_stage(), t + 1)] += 1
        for _ in range(a, total):  # Untreated patients wait
            p = sorted_patients[j].pop()
            patients[p.get_stage()].remove(p)
            c -= cost(*p.get_stage())
            p.wait()
            patients[p.get_stage()].insert(0, p)
            states[(*p.get_stage(), t + 1)] += 1
        if redistribute:
            # Redistribute capacity if less patients than action allows (only necessary for OD)
            if j != 'OR' and a < action[j]:
                od_patients = sorted_patients['FC'] + sorted_patients['RC'] + sorted_patients['DC']
                od_patients.sort(key=lambda x: reward[x.get_stage()[0]] + cost(*x.get_stage()), reverse=True)
                remaining_capacity = min(len(od_patients), action[j] - a) * zeta[j, 'OD']
                for p in od_patients:
                    remaining_capacity -= zeta[p.get_stage()[0], 'OD']
                    if remaining_capacity < 0:
                        break
                    sorted_patients[p.get_stage()[0]].remove(p)
                    patients[p.get_stage()].remove(p)
                    true_action[p.get_stage()] += 1
                    p.treat()
                    if p.active:
                        patients[p.get_stage()].insert(0, p)
                        states[(*p.get_stage(), t + 1)] += 1
    return states, patients, true_action, c
