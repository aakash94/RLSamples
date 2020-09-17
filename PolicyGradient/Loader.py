import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CriticLoader(Dataset):

    def __init__(self, dframe):

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        self.vs_header = ['state', 'rtg']
        self.vs = dframe

    def __len__(self):
        return len(self.vs.index)

    def __getitem__(self, index):
        row = self.vs.iloc[index]
        x = row[self.vs_header[0]]
        y = row[self.vs_header[1]]

        if type(x) is tuple:
            x = list(x)
        if type(y) is tuple:
            y = list(y)

        x = torch.Tensor(x).to(self.device).float()
        y = torch.Tensor(np.asarray(y)).to(self.device).flatten().float()

        return x, y


class ActorLoader(Dataset):

    def __init__(self, dframe, baseline_dict):
        self.header = ['state', 'action', 'rtg']
        self.dframe = dframe.filter(self.header, axis=1)
        self.baseline_dict = baseline_dict

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

    def __len__(self):
        return len(self.dframe.index)

    def __getitem__(self, index):
        row = self.dframe.iloc[index]
        state = row[self.header[0]]
        action = row[self.header[1]]
        rtg = row[self.header[2]]

        baseline = self.baseline_dict[state]

        if type(state) is tuple:
            state = list(state)
        if type(action) is tuple:
            action = list(action)
        if type(rtg) is tuple:
            rtg = list(rtg)

        advantage = rtg - baseline
        state = torch.Tensor(state).to(self.device).float()
        action = torch.Tensor(action).to(self.device).float()
        value = torch.Tensor(np.asarray(advantage)).to(self.device).flatten().float()

        return state, action, value


if __name__ == '__main__':
    pass