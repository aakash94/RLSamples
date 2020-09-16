import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, n_ip, move_to_gpu=True):
        super(Critic, self).__init__()
        size = 64
        activation = nn.Tanh()
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(n_ip, size))
        self.mlp.append(activation)
        self.mlp.append(nn.Linear(size, size))
        self.mlp.append(activation)
        self.mlp.append(nn.Linear(size, 1)) # state value


        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        if move_to_gpu:
            self.to_device()

    def forward(self, x):

        x = x.to(self.device)
        for layer in self.mlp:
            x = layer(x)

        return x

    def save_model(self, save_name):
        self.to_cpu()
        save_path = "saves/" + save_name + ".chkpt"
        torch.save(self.state_dict(), save_path)
        self.to_device()

    def load_model(self, load_name):
        self.to_cpu()
        save_path = "saves/" + load_name + ".chkpt"
        self.load_state_dict(torch.load(save_path))
        self.to_device()

    def to_device(self):
        if self.use_gpu:
            self.cuda()

    def to_cpu(self):
        self.cpu()
