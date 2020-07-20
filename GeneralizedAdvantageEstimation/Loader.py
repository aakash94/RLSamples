import torch
from torch.utils.data import Dataset

import numpy as np

from Trajectory import Trajectory
from UnitStep import UnitStep


class CriticLoader(Dataset):

    def __init__(self, data_collected):
        self.dataset = data_collected

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        # dataset = [ ([],[]), ([],[]), ([],[]), ([].[]) ]
        # or
        # dataset = [ ((),()), ((),()), ((),()), (().()) ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if type(x) is tuple:
            x = list(x)
        if type(y) is tuple:
            y = list(y)

        x = torch.Tensor(x).to(self.device).float()
        y = torch.Tensor(np.asarray(y)).to(self.device).flatten().float()

        return x, y


class StatesLoader(Dataset):

    def __init__(self, trajectory):
        self.dataset = trajectory

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        # dataset = [ UnitStep, UnitStep, UnitStep, UnitStep ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        step = self.dataset[index]
        state = step.state
        x = torch.Tensor(state).to(self.device)
        return x


class ActorLoader(Dataset):

    def __init__(self, data_collected):
        self.dataset = data_collected

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        # dataset = [ (((State),(Action)) : Advantage),
        #             (((State),(Action)) : Advantage),
        #             (((State),(Action)) : Advantage,)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key, value = self.dataset[index]
        state = list(key[0])
        action = list(key[1])
        if type(state) is tuple:
            state = list(state)
        if type(action) is tuple:
            action = list(action)

        state = torch.Tensor(state).to(self.device).float()
        action = torch.Tensor(action).to(self.device).float()
        value = torch.Tensor(np.asarray(value)).to(self.device).flatten().float()

        # state = Tensor[]
        # action = Tensor[]
        # value = number
        return state, action, value


if __name__ == '__main__':
    import torch.utils.data.dataloader as dataloader

    batch_size = 32
    epochs = 10
    expert_loader = CriticLoader(data_collected=[])
    loader = dataloader.DataLoader(expert_loader, batch_size=batch_size, shuffle=True)

    # optimizer = optim.Adam(self.agent.parameters(), lr=learn_rate)
    optimizer = None
    import torch.nn as nn

    criterion = nn.MSELoss()
    total_loss = 0

    for e in range(epochs):
        sum_loss = 0.0
        count = 0
        for x, y in loader:
            optimizer.zero_grad()
            # output = self.agent(x)
            output = x
            loss = criterion(output, y)
            sum_loss += loss.item()
            count += 1
            loss.backward()
            optimizer.step()

        total_loss += (sum_loss / count)
