from small_test_problem.lib.params_small import *
from small_test_problem.lib.functions import cost
import numpy as np


def one(s):
    return list(s) + [1]


def two(s):
    # phi = np.zeros((9, 1))
    phi = np.zeros(10)
    phi[-1] = 1
    for j in J:
        u = U[j]
        phi[j * 3] = sum(s[(j * 3) + w] for w in range(0, u))
        phi[(j * 3) + 1] = s[(j * 3) + u]
        phi[(j * 3) + 2] = sum(s[(j * 3) + w] for w in range(u+1, 3))
    return np.concatenate((phi[:3], phi[4:]))


def three(s):
    phi = np.zeros(4)
    phi[-1] = 1
    for j in J:
        u = U[j]
        phi[j] = sum(cost(j, u, w) * s[(j * 3) + w] for w in range(W))
    return phi


def four(s):
    phi = np.zeros(4)
    phi[3] = 1
    for j in J:
        phi[j] = sum(s[(j * 3) + w] for w in range(W))
    return phi


def one_two(s):
    return np.concatenate((one(s), two(s)))[:-1]


def one_three(s):
    return np.concatenate((one(s), three(s)))[:-1]


def one_four(s):
    return np.concatenate((one(s), four(s)))[:-1]


def two_three(s):
    return np.concatenate((two(s), three(s)))[:-1]


def two_four(s):
    return np.concatenate((two(s), four(s)))[:-1]


def three_four(s):
    return np.concatenate((three(s), four(s)))[:-1]
