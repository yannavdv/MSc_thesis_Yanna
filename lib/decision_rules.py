from lib.functions import *


def highest_contribution(state):
    """Treat patient that would result in highest increase in contribution"""
    action = {i: 0 for i in idx}
    for r in R:
        r_patients = [(j, u, w) for j in r_queues[r] for u in U[j] for w in range(W[j, u])
                      for _ in range(int(state[j, u, w]))]
        r_patients.sort(key=lambda x: (reward[x[0]] + cost(*x)) / zeta[x[0], r], reverse=True)
        capacity = eta[r]
        for p in r_patients:
            capacity -= zeta[p[0], r]
            if capacity < 0:
                break
            action[p] += 1
    return action


def highest_cost(state):
    """Treat patient of highest cost"""
    action = {i: 0 for i in idx}
    for r in R:
        r_patients = [(j, u, w) for j in r_queues[r] for u in U[j] for w in range(W[j, u])
                      for _ in range(int(state[j, u, w]))]
        r_patients = [item for sublist in r_patients for item in sublist]
        r_patients.sort(key=lambda x: cost(*x.get_stage()), reverse=True)
        r_patients = r_patients[:eta[r]]
        for p in r_patients:
            action[p.get_stage()] += 1
    return action


def highest_cost_queue(state):
    """Treat patient from queue with highest total cost"""
    action = {i: 0 for i in idx}
    state_copy = {(j, u): [state[j, u, w] for w in range(W[j, u])] for j in J for u in U[j]}
    queue_costs = [{(j, u): sum((omega[j] * w / (u + 1)) * state[j, u, w] for w in range(W[j, u]))
                    for j in r_queues[r] for u in U[j]} for r in R]
    i = 0
    for r in R:
        for _ in range(eta[r]):
            if sum(queue_costs[i].values()) < 0.01:
                break
            j, u = max(queue_costs[i], key=queue_costs[i].get)
            if len(np.nonzero(state_copy[j, u])[0]) == 0:
                w = W[j, u] - 1
            else:
                w = np.max(np.nonzero(state_copy[j, u]))
            state_copy[j, u][w] -= 1
            queue_costs[i][j, u] -= (omega[j] * w / (u + 1))
            action[j, u, w] += 1
        i += 1
    return action


def longest_queue(state):
    """Treat patient from the longest queue"""
    action = {i: 0 for i in idx}
    state_copy = {(j, u): [state[j, u, w] for w in range(W[j, u])] for j in J for u in U[j]}
    queue_lens = [{(j, u): sum(state[j, u, w] for w in range(W[j, u]))
                   for j in r_queues[r] for u in U[j]} for r in R]
    i = 0
    for r in R:
        for _ in range(eta[r]):
            if sum(queue_lens[i].values()) < 0.01:
                break
            j, u = max(queue_lens[i], key=queue_lens[i].get)
            if len(np.nonzero(state_copy[j, u])[0]) == 0:
                w = W[j, u] - 1
            else:
                w = np.max(np.nonzero(state_copy[j, u]))
            state_copy[j, u][w] -= 1
            queue_lens[i][j, u] -= 1
            action[j, u, w] += 1
        i += 1
    return action


def split_cost(state):
    """Distribute available resource capacity over queues, proportional to relative cost of queue to other queues"""
    action = {i: 0 for i in idx}
    queue_costs = {(j, u): 0 for j in J for u in U[j]}
    for (j, u, w) in idx:
        queue_costs[j, u] += (omega[j] * w / (u + 1)) * state[j, u, w]
    for r in R:
        total_cost = sum(queue_costs[j, u] for j in r_queues[r] for u in U[j])
        for j in r_queues[r]:
            for u in U[j]:
                if total_cost == 0:
                    remaining = 0
                else:
                    remaining = int(min(sum(state[j, u, w] for w in range(W[j, u])),
                                        (eta[r] * queue_costs[j, u]) // total_cost))
                for w in range(W[j, u] - 1, -1, -1):
                    a = min(state[j, u, w], remaining)
                    action[j, u, w] = a
                    remaining = remaining - a
    return action
