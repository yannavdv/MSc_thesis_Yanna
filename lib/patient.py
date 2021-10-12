from lib.parameters.params_smk import W
# from lib.parameters.params_large import W


class Patient:
    def __init__(self, path, stage, waiting_time):
        self.path = path  # Patient pathway
        self.stage = stage  # Integer indicating at which point the patient is in their pathway
        self.waiting_time = waiting_time  # Current waiting time of patient
        self.active = True  # False when patient is done with treatment

    def __str__(self):
        return 'Path: ' + str(self.path) + ' \nStage: ' + str(self.stage) + ' \nWaiting time: ' + \
               str(self.waiting_time) + ' \nActive: ' + str(self.active)

    def get_stage(self):
        return self.path[self.stage][0], self.path[self.stage][1], self.waiting_time

    def treat(self):
        self.stage += 1  # go to next stage in path
        self.waiting_time = 0
        if self.stage >= len(self.path):  # patient has completed treatment
            self.active = False

    def wait(self):
        self.waiting_time = min(self.waiting_time + 1, W[self.path[self.stage][0], self.path[self.stage][1]] - 1)


class ShortPatient:
    def __init__(self, path, stage, waiting_time):
        self.current = (path[stage][0], path[stage][1], waiting_time)
        if stage == len(path) - 1:
            self.next = 'Exit'
        else:
            self.next = (path[stage + 1][0], path[stage + 1][1], 0)

    def __str__(self):
        return 'Current: ' + str(self.current) + ' \nNext: ' + str(self.next)
