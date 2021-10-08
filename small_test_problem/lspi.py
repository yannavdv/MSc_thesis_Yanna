from small_test_problem.lib.functions import *
import pickle


def lspi(D, M, N, gamma):
    eps = 0.01
    delta = 0.01
    F = len(phi(D[0][0]))
    theta = np.zeros((F, 1))
    theta_dif = 100

    n = 0
    while (theta_dif > delta) and (n < N):
        n += 1
        if n == N:
            print('failed')
        # Initialise parameters
        b = np.zeros((F, 1))
        A = eps * np.eye(F)

        for m in range(M):
            s, patients = D[m]
            action = pi(s, theta, gamma)

            # Transition to new state
            s_new = np.zeros(len(J) * W)
            for j in J:
                for w in waiting_times:
                    a = action[(j * W) + w]
                    for i in range(a):
                        p = patients[(j, w)][i]
                        if p.next != 'Exit':
                            s_new[p.next * W] += 1
                    s_new[(j * W) + min(w + 1, W - 1)] += s[(j * W) + w] - a  # Untreated patients wait
            s_new[0] += num_new_patients

            # Recalculate theta
            b += contribution(s, action) * phi(s)
            A += np.matmul(phi(s), (phi(s) - gamma * phi(s_new)).T)

        theta_new = np.matmul(np.linalg.inv(A), b)
        theta_dif = np.linalg.norm(theta - theta_new)
        theta = theta_new
        print('Difference: ', theta_dif)
        for i in range(F):
            print(str(theta[i][0]), end=', ')
        print()
    return theta


M = 5000
D = generate_dataset(M, 60)
N = 10
gamma = 0.9

theta = lspi(D, M, N, gamma)
theta_file = open('lib/theta', 'wb')
pickle.dump(theta, theta_file)
theta_file.close()
