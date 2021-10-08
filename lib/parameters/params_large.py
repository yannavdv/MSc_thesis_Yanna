J = ['FC', 'RC', 'OR', 'DC']
R = ['OD', 'OR']
r_queues = {'OD': ['FC', 'RC', 'DC'], 'OR': ['OR']}

U = {'FC': [2], 'RC': [4], 'OR': [2, 4], 'DC': [3]}
omega = {'FC': 2, 'RC': 1, 'OR': 4, 'DC': 1}
reward = {'FC': 2, 'RC': 2, 'OR': 10, 'DC': 1}
num_new_patients = 8
eta = {'OD': 16, 'OR': 2}
W = {(j, u): 3 * u for j in J for u in U[j]}

zeta = {(j, r): 0 for j in J for r in R}
zeta[('FC', 'OD')] = 1
zeta[('RC', 'OD')] = 1
zeta[('DC', 'OD')] = 1
zeta[('OR', 'OR')] = 1

idx = [(j, u, w) for j in J for u in U[j] for w in range(W[j, u])]

q = {(i, u, j, v): 0 for i in J for u in U[i] for j in J for v in U[j]}

q[('FC', U['FC'][0], 'RC', U['RC'][0])] = 0.5
q['FC', U['FC'][0], 'OR', U['OR'][0]] = 0.01
q['FC', U['FC'][0], 'OR', U['OR'][1]] = 0.1

q[('RC', U['RC'][0], 'RC', U['RC'][0])] = 0.4
q[('RC', U['RC'][0], 'OR', U['OR'][0])] = 0.02
q[('RC', U['RC'][0], 'OR', U['OR'][1])] = 0.15

q[('OR', U['OR'][0], 'RC', U['RC'][0])] = 0.2
q[('OR', U['OR'][0], 'DC', U['DC'][0])] = 0.75

q[('OR', U['OR'][1], 'RC', U['RC'][0])] = 0.25
q[('OR', U['OR'][1], 'DC', U['DC'][0])] = 0.7

q[('DC', U['DC'][0], 'RC', U['RC'][0])] = 0.6
