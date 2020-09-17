import pandas as pd
import numpy as np
import random


class Episode:

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.header = ['state',
                       'action',
                       'reward',
                       'next_state',
                       'done',
                       'rtg', ]
        self.trajectory = pd.DataFrame(columns=self.header)
        self.row_list = []

    def __len__(self):
        return len(self.trajectory.index)

    def add_step(self,
                 state,
                 action,
                 reward,
                 next_state,
                 done=False,
                 rtg=0):
        values = [tuple(state), tuple(action), reward, tuple(next_state), done, rtg]
        d = dict(zip(self.header, values))
        self.row_list.append(d)

        if done:
            self.trajectory = pd.DataFrame(self.row_list)
            self.row_list = []
            self.compute_rtg()

    def add_random(self, state_size=3, action_size=2, done=False):
        state = np.random.rand(state_size)
        next_state = np.random.rand(state_size)
        action = np.random.rand(action_size)
        reward = random.randint(0, 100)
        rtg = 0  # random.randint(0, 100)
        values = [state, action, reward, next_state, done, rtg]
        d = dict(zip(self.header, values))
        self.row_list.append(d)

        if done:
            self.trajectory = pd.DataFrame(self.row_list)
            self.row_list = []
            self.compute_rtg()

    def compute_rtg(self):
        self.trajectory['rtg'] = self.trajectory['reward']
        length = self.__len__()
        for idx in reversed(self.trajectory.index):
            if idx + 1 >= length:
                continue
            self.trajectory.loc[idx, 'rtg'] += (self.trajectory.loc[idx + 1, 'rtg'] * self.gamma)


if __name__ == '__main__':
    pass