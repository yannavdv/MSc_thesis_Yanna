from lib.functions import *
import pickle

# LSPI algorithm (Section 7.2, Algorithm 6)

# Parameters (fill in)
M = 50000  # Dataset size
mu = 700  # Mean number of patients in each state
sigma = 200  # Variance of number of patients in each state
gamma = 0.6  # Discount factor
N = 10  # Maximum number of iterations
eps = 0.01
delta = 1.5

D = generate_dataset(M, mu, sigma)
F = len(phi(D[0][0]))
theta = np.zeros((F, 1))
theta_dif = 100

n = 0
while (theta_dif > delta) and (n < N):
    n += 1
    print(n)

    # Initialise parameters
    A = eps * np.eye(F)
    b = np.zeros((F, 1))

    for m in range(M):
        if m % 1000 == 0:
            print(m)
        s, patients = D[m]

        action = pi(s, theta, gamma)  # Use current policy to choose action

        # Transition to new state
        s_new = {i: 0 for i in idx}
        for (j, u, w) in idx:
            a = action[(j, u, w)]
            for i in range(a):
                p = patients[(j, u, w)][i]
                if p.next != 'Exit':
                    s_new[p.next] += 1
            s_new[(j, u, min(w + 1, W[j, u] - 1))] += s[(j, u, w)] - a  # Untreated patients wait
        # Add new patients
        for _ in range(num_new_patients):
            p = random.choice(all_paths)
            s_new[(*p[0], 0)] += 1

        # Recalculate theta
        b += contribution(s, action) * phi(s)
        A += np.matmul(phi(s), (phi(s) - gamma * phi(s_new)).T)

    theta_new = np.matmul(np.linalg.inv(A), b)
    theta_dif = np.linalg.norm(theta - theta_new)
    theta = theta_new
    print('Difference: ', theta_dif)

# Save parameter vector for use in simulation
theta_file = open('lib/theta', 'wb')
pickle.dump(theta, theta_file)
theta_file.close()
