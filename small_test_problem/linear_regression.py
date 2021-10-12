from small_test_problem.lib.basis_functions import *
import pickle
from sklearn.model_selection import KFold
import random

# Parameters (fill in)
k = 10  # number of folds for k-fold cross validation
phi = one  # basis function to test (see lib/basis_functions.py)

F = len(phi(list(states[0])))
function_name = phi.__name__

# Make train & test sets
random.shuffle(states)
dataset = np.array(states)
dataset_size = len(dataset)
kf = KFold(n_splits=k)
errors = []

# Load value function
v_file = open("lib/value_function.pkl", "rb")
v = pickle.load(v_file)
v_file.close()

f_size = open('linreg_' + function_name + '_size.dat', 'w')
f_size.write('size error\n')
f_ub = open('linreg_' + function_name + '_ub.dat', 'w')
f_ub.write('ub error\n')

size_error = [0] * 34
size_counter = [0] * 34
ub_error = [0] * 10
ub_counter = [0] * 10

for train, test in kf.split(dataset):
    train_states = dataset[train]
    test_states = dataset[test]
    train_size = len(train_states)

    # Train
    X = np.zeros((train_size, F))
    Y = np.zeros(train_size)
    for i in range(train_size):
        s = list(train_states[i])
        Y[i] = v[tuple(s)]
        xfre = phi(train_states[i])
        X[i] = phi(train_states[i])
    theta = np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), np.matmul(X.transpose(), Y))
    print('Theta: ' + str(theta))

    # Test
    mse = (1 / len(test_states)) * sum((v[tuple(s)] - np.dot(theta, phi(s)))**2 for s in test_states)
    print('MSE: ', mse)
    errors += [mse]
    for s in test_states:
        s = list(s)
        size_state = sum(s)
        ub = sum([s[(j * 3) + w] == l[j] for j in J for w in range(W)])
        error = abs(v[tuple(s)] - np.dot(theta, phi(s)))
        size_error[size_state] += error
        size_counter[size_state] += 1
        ub_error[ub] += error
        ub_counter[ub] += 1

errors = np.array(errors)
print('Average MSE: ', np.mean(errors))
print('Std dev MSE: ', np.std(errors))

for i in range(10):
    if ub_counter[i] > 0:
        f_ub.write(str(i) + ' ' + str(ub_error[i] / ub_counter[i]) + '\n')

for i in range(34):
    if size_counter[i] > 0:
        f_size.write(str(i) + ' ' + str(size_error[i] / size_counter[i]) + '\n')

f_size.close()
f_ub.close()
