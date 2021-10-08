from lib.simulation_functions import *
from copy import deepcopy

# Fill in
trials = 100
time = 16
filename = 'num_patients_large'
num_patients_range = list(range(400, 900, 50))

theta_file = open('lib/theta', 'rb')
theta = pickle.load(theta_file)
theta_file.close()

algos = ['LSPI', 'LP', 'HighestContribution', 'Standard']
gamma = {'LSPI': 0.6, 'LP': 0.75, 'HighestContribution': 1, 'Standard': 0.5}
n = len(num_patients_range)
costs = {a: [] for a in algos}

f = open(filename + '.dat', 'w')
f.write('P ')
for a in algos:
    f.write(a + ' ')
f.write('\n')

for num_initial_patients in num_patients_range:
    print(num_initial_patients)
    sums = {a: 0 for a in algos}
    for trial in range(trials):
        states, patients, new_patients = init_state(num_initial_patients, time, num_new_patients)
        for a in algos:
            sums[a] += simulate_solution(deepcopy(states), deepcopy(patients), deepcopy(new_patients), time,
                                         algo=a, theta=theta, gamma=gamma[a])
    f.write(str(num_initial_patients) + ' ')
    for a in algos:
        costs[a] += [sums[a] / trials]
        f.write(str(sums[a] / trials) + ' ')
    f.write('\n')

f.close()

# Plot results
fig = go.Figure(
    data=[go.Scatter(x=num_patients_range, y=costs[a], name=a) for a in algos],
    layout=go.Layout(
        title=go.layout.Title(text="Performance of solution methods on full test problem")
    )
)
fig.update_xaxes(title_text="Number of initial patients")
fig.update_yaxes(title_text="Average contribution")
fig.show()
