import numpy as np
import pandas as pd
from Episode import Episode
from collections import defaultdict


class Runs:

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.ep_header = ['state', 'action', 'reward', 'next_state', 'done', 'rtg', ]
        self.vs_header = ['state', 'rtg']
        self.qsa_header = ['sa', 'rtg']
        self.reset()

    def reset(self):
        self.trajectories = []

        # for whatever trajectory is added first here
        self.episode = Episode(gamma=self.gamma)
        self.qsa = pd.DataFrame(columns=self.qsa_header)

        self.v_s_rounded = {}
        # self.q_sa = {}

    def add_step(self, state, action, reward, next_state, done):
        self.episode.add_step(state=state,
                              action=action,
                              reward=reward,
                              next_state=next_state,
                              done=done)
        if done:
            self.trajectories.append(self.episode.trajectory)
            self.episode = Episode(gamma=self.gamma)

    def compute_rewards(self, round_decimals=3):
        # merged_df = pd.concat(self.episodes)
        merged_df = pd.concat(self.trajectories)
        tiled = merged_df.filter(self.vs_header, axis=1)
        tiled[self.vs_header[0]] = tiled[self.vs_header[0]].apply(lambda x: tuple(np.round(x, round_decimals)))
        tiled = tiled.groupby(self.vs_header[0])[self.vs_header[1]].mean()
        self.v_s_rounded = tiled.to_dict()
        # groupby mean of tiled here
        # qsa = pd.DataFrame(columns=self.qsa_header)
        self.qsa['sa'] = list(zip(merged_df['state'], merged_df['action']))
        self.qsa['rtg'] = merged_df['rtg'].values
        self.qsa = self.qsa.groupby(self.qsa_header[0])[self.qsa_header[1]].mean().reset_index()
        # print(self.qsa)
        # self.q_sa = qsa.to_dict()


if __name__ == '__main__':
    pass
