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
        len = self.__len__()
        for idx in reversed(self.trajectory.index):
            if idx + 1 >= len:
                continue
            # self.trajectory.rtg[idx] += (self.trajectory.rtg[idx + 1] * self.gamma)
            self.trajectory.loc[idx, 'rtg'] += (self.trajectory.loc[idx + 1, 'rtg'] * self.gamma)


if __name__ == '__main__':
    e = Episode(gamma=1)
    time_steps = 10
    for i in range(time_steps):
        if i == time_steps - 1:
            done = True
        else:
            done = False
        e.add_random(done=done)

    # print(e.trajectory.to_string())
    vs_header = ['state', 'action', 'rtg']
    # tiled = e.trajectory.filter(vs_header, axis=1)
    tiled = e.trajectory.filter(vs_header, axis=1)
    # print(tiled.to_string())

    # print(type(tiled))
    # tiled.round(decimals=2)
    # tiled['state'] = tiled['state'].apply(lambda x:tuple(np.round(x, 3)))
    tiled['state'] = tiled['state'].apply(lambda x: np.round(x, 3))
    tiled['action'] = tiled['action'].apply(lambda x: np.round(x, 3))
    # tiled['action'] = tiled['action'].apply(lambda x:tuple(x))
    #
    # tiled['state'].loc[0] = (0.010, 0.012, 0.014)
    # tiled['state'].loc[3] = (0.010, 0.012, 0.014)
    # tiled['state'].loc[6] = (0.010, 0.012, 0.014)
    # tiled['state'].loc[9] = (0.010, 0.012, 0.014)
    #
    #
    #
    # print(tiled.to_string())
    #
    # tiled = tiled.groupby('state')['rtg'].mean()
    # print(tiled.to_string())
    # a = tiled.to_dict()
    # print(a)
    #
    # print(tiled[['state','action']])
    # print(tiled)

    # a = list(zip(tiled['state'], tiled['action']))
    # for i in a:
    #     print(i)
    # df['new_col'] = list(zip(df.lat, df.long))

    qsa_header = ['sa', 'rtg']
    qsa = pd.DataFrame(columns=qsa_header)
    qsa['sa'] = list(zip(tiled['state'], tiled['action']))
    qsa['rtg'] = tiled['rtg']
    print(qsa)

    row = qsa.iloc[2]
    sa = row['sa']
    s =sa[0]
    a = sa[1]
    r = row['rtg']
    print(s)
    print(a)
    print(r)
