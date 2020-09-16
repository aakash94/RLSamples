import torch
import numpy as np
import pandas as pd
from Episode import Episode
from Loader import CriticLoader
from Critic import Critic
import torch.utils.data.dataloader as dataloader

class Runs:

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.ep_header = ['state', 'action', 'reward', 'next_state', 'done', 'rtg', ]
        self.vs_header = ['state', 'rtg']
        #self.qsa_header = ['sa', 'rtg']
        self.reset()

    def reset(self):
        self.trajectories = []
        self.baseline = {}

        # for whatever trajectory is added first here
        self.episode = Episode(gamma=self.gamma)
        self.all_runs = pd.DataFrame(columns=self.ep_header)
        #self.qsa = pd.DataFrame(columns=self.qsa_header)
        self.vs = pd.DataFrame(columns=self.vs_header)

    def add_step(self, state, action, reward, next_state, done):
        self.episode.add_step(state=state,
                              action=action,
                              reward=reward,
                              next_state=next_state,
                              done=done)
        if done:
            self.trajectories.append(self.episode.trajectory)
            self.episode = Episode(gamma=self.gamma)

    def compute_rewards(self):
        self.all_runs = pd.concat(self.trajectories)
        self.vs = self.all_runs.filter(self.vs_header, axis=1).copy()

        #self.qsa['sa'] = list(zip(merged_df['state'], merged_df['action']))
        #self.qsa['rtg'] = merged_df['rtg'].values

    def get_normalized_rtg(self):
        df_norm = self.vs.copy()
        m = df_norm['rtg'].mean()
        s = df_norm['rtg'].std()

        df_norm['rtg'] = (df_norm['rtg'] - m) / s
        return df_norm

    def compute_baseline_dict(self, critic, batch_size=1024):
        with torch.no_grad():
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            mean_val = self.vs['rtg'].mean()
            std_val = self.vs['rtg'].std()
            length = len(self.all_runs.index)

            start_row = 0
            while start_row < length:
                end_row = start_row + batch_size
                end_row = min(end_row, length)
                chunk = self.all_runs[start_row:end_row]
                states = chunk['state'].to_list()
                targets = chunk['rtg'].to_list()

                t_states = torch.Tensor(states).to(device)
                t_targets = torch.Tensor(targets).to(device).flatten().float()
                t_baselines = critic(t_states)
                t_baselines = t_baselines.flatten()
                t_baselines = t_baselines * std_val + mean_val
                advantage = t_targets - t_baselines
                advantage = advantage.to("cpu").numpy()

                for index, state in enumerate(states):
                    adv = advantage[index]
                    s = tuple(state)
                    self.baseline[s] = adv


                start_row = end_row

            '''
            data_loader = CriticLoader(dframe=self.vs)
            loader = dataloader.DataLoader(data_loader, batch_size=batch_size, shuffle=True)
            for states, targets in loader:
                baseline = critic(states)
                baseline = baseline.flatten()
                baseline = baseline*std_val + mean_val
                advantage = targets - baseline
                states = states.to("cpu")
                advantage = advantage.to("cpu")

                # add state and adbantage to dictionary
                for index, state in enumerate(states):
                    adv = advantage[index]
                    s = tuple(state.numpy())
                    self.baseline[s] = adv
            '''


if __name__ == '__main__':
    vs_header = ['state', 'rtg']
    df_norm = pd.DataFrame(columns=vs_header)
    row_list = []
    state = np.array([1, 2, 3])
    rtg = 99
    values = [tuple(state), rtg]
    d = dict(zip(vs_header, values))
    row_list.append(d)

    state = np.array([4, 5, 6])
    rtg = 98
    values = [tuple(state), rtg]
    d = dict(zip(vs_header, values))
    row_list.append(d)
    state = np.array([7, 8, 9])
    rtg = 97
    values = [tuple(state), rtg]
    d = dict(zip(vs_header, values))
    row_list.append(d)
    df_norm = pd.DataFrame(row_list)
    print(df_norm)

    m = df_norm['rtg'].mean()
    s = df_norm['rtg'].std()
    print(m)
    print(s)
    df_norm['rtg'] = (df_norm['rtg'] - df_norm['rtg'].mean()) / df_norm['rtg'].std()
    print(df_norm)
