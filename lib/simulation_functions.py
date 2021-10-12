from lib.functions import *
from lib.lp import solve_lp
from lib.decision_rules import highest_contribution as decision_rule  # Change to other decision rule if required
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_solution(time_horizon, states, actions, C):
    """
    Generates plot of patient queues and actions throughout simulation
    :param time_horizon: Time horizon of simulation
    :param states: States throughout simulation
    :param actions: Actions performed by solution method in simulation
    :param C: Contribution per time period in simulation
    :return: Plot showing states, actions, contribution and excess OD capacity throughout simulation
    """
    x = list(range(0, time_horizon * 2, 2))
    times = list(range(time_horizon))
    early = {(j, t): 0 for j in J for t in times}
    late = {(j, t): 0 for j in J for t in times}
    excess = [0] * time_horizon
    for t in range(time_horizon):
        excess[t] = eta['OD'] - sum((zeta[j, 'OD'] * actions[(j, u, w, t)])
                                    for j in r_queues['OD'] for u in U[j] for w in range(W[j, u]))
        for j in J:
            late[j, t] = sum(states[j, u, w, t] for u in U[j] for w in range(u + 1, W[j, u]))
            early[j, t] = sum(states[j, u, w, t] for u in U[j] for w in range(u + 1))

    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=('First consultation (FC)', 'Repeat consultation (RC)', 'Surgery (OR)',
                                        'Discharge consultation (DC)', 'Total objective value = ' + str(sum(C)),
                                        'Excess OD capacity'))
    fig_idx = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for i in range(4):
        j = J[i]
        fig.add_trace(go.Scatter(x=x, y=[early[j, t] for t in times], name='Early patients',
                                 line=dict(color='teal'), legendgroup='Early patients', showlegend=(i == 0)),
                      row=fig_idx[i][0], col=fig_idx[i][1])
        fig.add_trace(go.Scatter(x=x, y=[late[j, t] for t in times], name='Late patients',
                                 line=dict(color='tomato'), legendgroup='Late patients', showlegend=(i == 0)),
                      row=fig_idx[i][0], col=fig_idx[i][1])
        fig.add_trace(go.Scatter(x=x, y=[sum(actions[j, u, w, t] for u in U[j] for w in range(W[j, u])) for t in times],
                                 name='Number of patients treated', line=dict(color='gold'),
                                 legendgroup='Number of patients treated', showlegend=(i == 0)),
                      row=fig_idx[i][0], col=fig_idx[i][1])
        fig.update_xaxes(title_text="Time (weeks)", row=fig_idx[i][0], col=fig_idx[i][1])
        fig.update_yaxes(title_text="Number of patients", row=fig_idx[i][0], col=fig_idx[i][1])

    fig.add_trace(go.Scatter(x=x, y=C, name='Objective', line=dict(color='mediumvioletred')), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=excess, name='Excess', line=dict(color='mediumvioletred')), row=3, col=2)
    return fig


def simulate_solution(states, patients, new_patient_list, time_horizon, algo='LP', theta=None, gamma=0.75, P=0,
                      redistribute=False, alpha=0, plot=False):
    """
    Simulation of patient treatment
    :param states: Initial state of system
    :param patients: Patients initially in system
    :param new_patient_list: Patients to add to system during simulation
    :param time_horizon: Number of time periods to run simulation
    :param algo: Which algorithm to use (LP, LSPI, Decision_Rule, Static)
    :param theta: Parameter vector for LSPI
    :param gamma: Discount factor for LSPI and LP; percentage of capacity allocated to FC appointments
                  for Static roster method
    :param P: Planning horizon
    :param redistribute:
    :param alpha: Percentage of appointments to fix in the LP when using static roster for Hybrid method
    :param plot: If True: generates plot of patient queues and actions throughout simulation
    :return: Average contribution per time period (unless plot=True, in which case the plot is returned)
    """
    actions = {(*i, t): 0 for i in idx for t in range(time_horizon)}
    compact_actions = {(j, t): 0 for j in J for t in range(time_horizon)}
    C = [0] * time_horizon
    for t in range(time_horizon):
        if P > 0 and t > 0:
            # Determine what the predicted state would be in this time period
            # if the prediction was made P time period ago
            current_state = predict_state({i: states[(*i, max(t-P, 0))] for i in idx},
                                          compact_actions, max(t-P, 0), t)
        else:
            # No prediction required: plan for the current state
            current_state = {i: states[(*i, t)] for i in idx}
        # Determine action using solution method specified by algo
        if algo == 'LP':
            action = solve_lp(current_state, 10, gamma, alpha=alpha)
        elif algo == 'LSPI':
            action = pi(current_state, theta, gamma)
        elif algo == 'Decision_Rule':
            action = decision_rule(current_state)
        elif algo == 'Static':
            action = {j: 0 for j in J}
            action['DC'] = eta['OR']
            action['FC'] = int((eta['OD'] * gamma) // zeta['FC', 'OD'])
            action['RC'] = eta['OD'] - action['DC'] - (zeta['FC', 'OD'] * action['FC'])
            action['OR'] = eta['OR']
        else:
            print('Invalid algorithm given')
            return 0

        # Perform action selected by solution method
        if (P > 0 and t > 0) or algo == 'Static':  # Treat patients using a compact action
            if algo != 'Static':
                action = {j: int(sum(action[j, u, w] for u in U[j] for w in range(W[j, u]))) for j in J}
            for j in J:
                compact_actions[j, t] = action[j]
            states, patients, true_action, C[t] = perform_compact_action(states, patients, t, action, redistribute)
            for i in idx:
                actions[(*i, t)] = true_action[i]

        else:  # No planning ahead; perform action as specified
            for (j, u, w) in idx:
                a = int(action[j, u, w])
                actions[j, u, w, t] = a
                C[t] += a * reward[j]
                for i in range(a):  # Treat specified number of patients
                    p = patients[j, u, w].pop()
                    p.treat()
                    if p.active:
                        patients[p.get_stage()].insert(0, p)
                        states[(*p.get_stage(), t + 1)] += 1
                for i in range(a, current_state[j, u, w]):  # Untreated patients wait
                    p = patients[j, u, w].pop()
                    C[t] -= cost(j, u, w)
                    p.wait()
                    patients[p.get_stage()].insert(0, p)
                    states[(*p.get_stage(), t + 1)] += 1

        # Add new patients to system
        for _ in range(num_new_patients):
            p = new_patient_list.pop()
            patients[p.get_stage()].insert(0, p)
            states[(*p.get_stage(), t + 1)] += 1

    if plot:
        return plot_solution(time_horizon, actions, states, C)
    else:
        return sum(C[P + 1:]) / (time_horizon - (P + 1))
