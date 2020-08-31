import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Loader(Dataset):

    def __init__(self, vs_round, qsa, round_decimals=3):

        self.v_s_rounded = vs_round
        self.qsa = qsa
        self.round_decimals = round_decimals

        #qsa_header = ['sa', 'rtg']
        #self.qsa = pd.DataFrame(columns=qsa_header)

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

    def __len__(self):
        return len(self.qsa.index)

    def __getitem__(self, index):
        row = self.qsa.iloc[index]
        sa = row['sa']
        state = sa[0]
        action = sa[1]
        rtg = row['rtg']
        rounded_state = tuple(np.round(state, self.round_decimals))
        baseline_value = self.v_s_rounded[rounded_state]
        value = rtg - baseline_value

        state = torch.Tensor(state).to(self.device).float()
        action = torch.Tensor(action).to(self.device).float()
        value = torch.Tensor(np.asarray(value)).to(self.device).flatten().float()

        return state, action, value
