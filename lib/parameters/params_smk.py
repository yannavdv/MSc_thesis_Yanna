import pickle

J = ['FC', 'RC', 'OR', 'DC']
R = ['OD', 'OR']
r_queues = {'OD': ['FC', 'RC', 'DC'], 'OR': ['OR']}

U = {'FC': [2], 'RC': [3, 6, 12], 'OR': [1, 2, 4, 6], 'DC': [3]}
omega = {'FC': 0.5, 'RC': 3, 'OR': 10, 'DC': 3}
reward = {'FC': 5, 'RC': 3, 'OR': 50, 'DC': 3}
W = {(j, u): min(3 * u, 18) for j in J for u in U[j]}

zeta = {(j, r): 0 for j in J for r in R}
zeta[('FC', 'OD')] = 2
zeta[('RC', 'OD')] = 1
zeta[('OR', 'OR')] = 1
zeta[('DC', 'OD')] = 1

T = 40
idx = [(j, u, w) for j in J for u in U[j] for w in range(W[j, u])]

num_new_patients = 40
eta = {'OD': 121, 'OR': 9}

q_file = open('lib/parameters/transition_probabilities', 'rb')
q = pickle.load(q_file)
q_file.close()

lambdas_file = open('lib/parameters/lambda', 'rb')
lambdas = pickle.load(lambdas_file)
lambdas_file.close()

paths_file = open('lib/parameters/all_paths', 'rb')
all_paths = pickle.load(paths_file)
paths_file.close()
