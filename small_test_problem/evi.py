from small_test_problem.lib.functions import *
from math import comb
import pickle

gamma = 0.9
eps = 0.1
delta = eps * (1 - gamma) / (2 * gamma)

v = {s: max([contribution_evi(pre_transition(list(s).copy(), a)[0], a) for a in possible_actions(s)]) for s in states}
norm_v_dif = 1000
norm_file = open('norms_exact.dat', 'w')
norm_file.write('iteration norm_v_dif\n')
n = 1

while norm_v_dif > delta:
    print('Iteration ' + str(n))
    v_new = {s: None for s in states}

    for s in states:
        s = list(s)
        max_a = -1000
        actions = possible_actions(s)
        for a in actions:
            s_pre, s_before_wait = pre_transition(s.copy(), a)
            C = contribution_evi(s_before_wait, a)
            sum_x = 0
            for x in possible_transitions(a):
                s_new = tuple(transition(s_pre, x))
                if v_new[s_new] is not None:
                    v_t = v_new[s_new]
                else:
                    v_t = v[s_new]
                sum_x += np.prod([comb(a[i], x[(i, j)]) * (q[(i, j)] ** x[(i, j)]) * ((1-q[(i, j)]) ** (a[i]-x[(i, j)]))
                                  for (i, j) in I]) * v_t
            max_a = max(max_a, C + (gamma * sum_x))
        v_new[tuple(s)] = max_a

    v_dif = {s: v_new[s] - v[s] for s in states}
    v = v_new.copy()
    norm_v_dif = norm(v_dif)
    print('Norm difference: ' + str(norm_v_dif))
    norm_file.write(str(n) + ' ' + str(norm_v_dif) + '\n')
    n += 1

norm_file.close()
value_file = open('value_function.pkl', 'wb')
pickle.dump(v, value_file)
value_file.close()
