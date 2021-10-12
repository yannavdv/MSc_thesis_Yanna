from small_test_problem.lib.small_lp import solve_lp
from small_test_problem.lib.functions import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from math import comb

v_file = open("lib/value_function.pkl", "rb")
v = pickle.load(v_file)
v_file.close()


def plot_solution(time, actions, states, C):
    x = list(range(0, time * 2, 2))
    times = list(range(time))
    early = {(j, t): 0 for j in J for t in times}
    late = {(j, t): 0 for j in J for t in times}
    for t in range(time):
        for j in J:
            late[j, t] = sum(states[j, w, t] for w in range(U[j] + 1, W))
            early[j, t] = sum(states[j, w, t] for w in range(U[j] + 1))

    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=('First consultation (FC)', 'Repeat consultation (RC)', 'Surgery 0 (OR_0)',
                                        'Surgery 1 (OR_1)', 'Excess OD capacity',
                                        'Total objective value = ' + str(sum(C))))
    idx = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for j in J:
        fig.add_trace(go.Scatter(x=x, y=[early[j, t] for t in times], name='Early patients', line=dict(color='teal'),
                                 legendgroup='Early patients', showlegend=(j == 0)), row=idx[j][0], col=idx[j][1])
        fig.add_trace(go.Scatter(x=x, y=[late[j, t] for t in times], name='Late patients', line=dict(color='tomato'),
                                 legendgroup='Late patients', showlegend=(j == 0)), row=idx[j][0], col=idx[j][1])
        fig.add_trace(go.Scatter(x=x, y=[actions[j, t] for t in times], name='Number of patients treated',
                                 line=dict(color='gold'), legendgroup='Number of patients treated',
                                 showlegend=(j == 0)), row=idx[j][0], col=idx[j][1])
        fig.update_xaxes(title_text="Time (weeks)", row=idx[j][0], col=idx[j][1])
        fig.update_yaxes(title_text="Number of patients", row=idx[j][0], col=idx[j][1])

    excess_od = [eta['OD'] - actions[0, t] - actions[1, t] for t in times]
    fig.add_trace(go.Scatter(x=x, y=excess_od, name='Excess OD capacity', line=dict(color='darkblue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=C, name='Objective', line=dict(color='mediumvioletred')), row=3, col=2)
    return fig


def simulate_solution(states, patients, new_patient_list, time, algo='LP', theta=None, gamma=0.9, plot=False):
    actions = {(j, t): 0 for t in range(time) for j in J}
    C = [0] * time
    for t in range(time):
        current_state = {(j, w): states[(j, w, t)] for j in J for w in range(W)}
        if algo == 'LP':
            action = solve_lp(current_state, time)

        elif algo == 'Exact':
            s = [min(current_state[j, w], l[j]) for j in J for w in waiting_times]
            max_a = -1000
            action_short = [0, 0, 0]
            # Get feasible actions
            for ac in possible_actions(s):
                s_pre, s_before_wait = pre_transition(s.copy(), ac)
                sum_x = 0
                for x in possible_transitions(ac):
                    s_new = tuple(transition(s_pre, x))
                    sum_x += np.prod(
                        [comb(ac[i], x[(i, j)]) * (q[(i, j)] ** x[(i, j)]) * ((1 - q[(i, j)]) ** (ac[i] - x[(i, j)]))
                         for (i, j) in I]) * v[s_new]
                cont = sum(ac[i] * reward[i] for i in J) - sum(cost(j, U[j], w) * s_before_wait[(j * 3) + w]
                                                               for j in J for w in waiting_times)
                if cont + (0.9 * sum_x) > max_a:
                    action_short = ac
                    max_a = cont + (0.9 * sum_x)

            action = [0] * len(J) * W
            for j in J:
                remaining = action_short[j]
                for w in range(W - 1, -1, -1):
                    a = min(current_state[j, w], remaining)
                    action[(j * W) + w] = a
                    remaining = remaining - a

        elif algo == 'LSPI':
            s = [current_state[j, w] for j in J for w in waiting_times]
            action = pi(s, theta, gamma)

        else:
            action = 0

        # Treat patients according to action
        for j in J:
            for w in range(W):
                a = action[(j * W) + w]
                actions[j, t] += a
                C[t] += a * reward[j]
                for _ in range(a):
                    p = patients[j, w].pop()
                    p.treat()
                    if p.active:
                        patients[p.get_stage(), 0].insert(0, p)
                        states[p.get_stage(), 0, t + 1] += 1
                for _ in range(a, current_state[j, w]):
                    p = patients[j, w].pop()
                    C[t] -= cost(j, U[j], w)
                    p.wait()
                    patients[p.get_stage(), p.waiting_time].insert(0, p)
                    states[p.get_stage(), p.waiting_time, t + 1] += 1

        patients[0, 0] += new_patient_list[:num_new_patients]
        states[0, 0, t + 1] += num_new_patients
        new_patient_list = new_patient_list[num_new_patients:]

    if plot:
        return plot_solution(time, actions, states, C)
    else:
        return sum(C)
