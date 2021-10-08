from small_test_problem.lib.simulation_functions import *
from copy import deepcopy

# Fill in
trials = 2
time = 30
num_patients_range = list(range(0, 31, 5))
gamma = 0.9
filename = 'test_num_patients_contribution'

algos = ['LP', 'LSPI', 'Exact']
costs = {a: [] for a in algos}
theta_file = open("lib/theta", "rb")
theta = pickle.load(theta_file)
theta_file.close()

f = open(filename + '.dat', 'w')
f.write('num_patients ')
for a in algos:
    f.write(a + ' ')
f.write('\n')

for num_initial_patients in num_patients_range:
    print(num_initial_patients)
    sums = {a: 0 for a in algos}

    for trial in range(trials):
        initial_patients = [generate_random_patient() for _ in range(num_initial_patients)]
        patients = {(j, w): [] for j in J for w in waiting_times}
        states = {(j, w, t): 0 for t in range(time+1) for j in J for w in waiting_times}

        for p in initial_patients:
            j = p.get_stage()
            w = p.waiting_time
            patients[j, w] += [p]
            states[j, w, 0] += 1

        new_patients = [generate_random_patient(new=True) for _ in range(time * num_new_patients)]

        for a in algos:
            sums[a] += simulate_solution(deepcopy(states), deepcopy(patients), deepcopy(new_patients), time, algo=a,
                                         theta=theta, gamma=gamma)

    f.write(str(num_initial_patients) + ' ')
    for a in algos:
        costs[a] += [sums[a] / trials]
        f.write(str(sums[a] / trials) + ' ')
    f.write('\n')

f.close()

fig = go.Figure(
    data=[go.Scatter(x=num_patients_range, y=costs[a], name=a) for a in algos],
    layout=go.Layout(
        title=go.layout.Title(text="Performance of algorithms on test problem for " + str(time) + " time periods")
    )
)
fig.update_xaxes(title_text="Number of initial patients")
fig.update_yaxes(title_text="Total contribution (average over " + str(trials) + " trials)")
fig.show()
