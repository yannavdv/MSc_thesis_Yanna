J = [0, 1, 2]
R = ['OD', 'OR']
r_queues = {'OD': [0], 'OR': [1, 2]}
U = {0: 1, 1: 0, 2: 1}
W = 3
waiting_times = range(W)
double_theta = 4
eta = {'OD': 4, 'OR': 2}
l = [7, 2, 2]  # upper bound on state size
q = {(0, 0): 0.3, (0, 1): 0.2, (0, 2): 0.3, (0, 3): 0.2,
     (1, 0): 0.4, (1, 3): 0.6, (2, 0): 0.4, (2, 3): 0.6}
I = list(q.keys())
q_full = {(i, j): 0 for i in J for j in J}
q_full.update(q)
reward = [1, 4, 4]
wait_cost = [1, 4, 4]
num_new_patients = 2

# Generate all possible states
rc_1 = [(a, b, c) for a in range(l[0]+1) for b in range(l[0]+1) for c in range(l[0]+1)]
or_0 = [(a, b, c) for a in range(l[1]+1) for b in range(l[1]+1) for c in range(l[1]+1)]
or_1 = [(a, b, c) for a in range(l[2]+1) for b in range(l[2]+1) for c in range(l[2]+1)]
states = [a + b + c for a in rc_1 for b in or_0 for c in or_1]

# # Generate all possible actions
all_capacities = [[a, b, c] for a in range(eta['OD'] + 1) for b in range(eta['OR'] + 1) for c in range(eta['OR'] + 1)]
all_actions = []
for [a, b, c] in all_capacities:
    if b + c <= eta['OR']:
        all_actions += [[a, b, c]]
