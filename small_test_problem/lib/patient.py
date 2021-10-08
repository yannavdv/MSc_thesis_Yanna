import numpy as np
from small_test_problem.lib.params_small import *


class Patient:
    def __init__(self, path, stage, waiting_time):
        self.path = path  # Patient pathway
        self.stage = stage  # Integer indicating at which point the patient is in their pathway
        self.waiting_time = waiting_time
        self.active = True  # False when patient is done with treatment
        self.late = (self.waiting_time > self.get_urgency())

    def __str__(self):
        return 'Path: ' + str(self.path) + ' \nStage: ' + str(self.stage) + ' \nWaiting time: ' + \
               str(self.waiting_time) + ' \nActive: ' + str(self.active) + ' \nLate: ' + str(self.late)

    def get_urgency(self):
        return U[self.path[self.stage]]

    def get_stage(self):
        return self.path[self.stage]

    def treat(self):
        self.stage += 1  # go to next stage in path
        self.waiting_time = 0
        self.late = False
        if self.stage >= len(self.path):  # patient has completed treatment
            self.active = False

    def wait(self):
        self.waiting_time = min(self.waiting_time + 1, W-1)
        if self.waiting_time > self.get_urgency():  # update lateness
            self.late = True


class ShortPatient:
    def __init__(self, path, stage, waiting_time):
        self.current = path[stage]
        self.waiting_time = waiting_time
        if stage == len(path) - 1:
            self.next = 'Exit'
        else:
            self.next = path[stage + 1]

    def __str__(self):
        return 'Current: ' + str(self.current) + ' \nNext: ' + str(self.next)


def generate_random_patient(new=False, short=False):
    # Decide starting queue
    path = [0]
    done = False
    while not done:
        current = path[-1]
        r = np.random.rand()
        total = 0
        for j in J:
            total += q_full[current, j]  # probability of transitioning to (j, u)
            if r <= total:
                path += [j]
                break
        if r > total:  # no transition took place
            done = True

    if new:  # patient has just entered system
        stage = 0
        waiting_time = 0
    else:  # patient is at some random place in their path
        stage = np.random.choice(range(len(path)))
        waiting_time = np.random.choice(W)
    if short:
        return ShortPatient(path, stage, waiting_time)
    else:
        return Patient(path, stage, waiting_time)


def treat_patients(untreated, num_patients):
    treated = []
    for _ in range(num_patients):
        p = untreated.pop(0)  # Highest waiting time is first in line
        p.treat()
        if p.active:
            treated.append(p)
    for p in untreated:
        p.wait()
    return untreated, treated
