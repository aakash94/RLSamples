import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    # For continuous actions
    # returning mean and standard deviation

    def __init__(self, n_ip, drop_p=0.2, move_to_gpu=True):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_ip, n_ip * 16)
        self.fc2 = nn.Linear(n_ip * 16, n_ip * 16)
        self.fc3 = nn.Linear(n_ip * 16, 1)

        self.dropout = nn.Dropout(p=drop_p)

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
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        t_value = self.fc3(x)

        return t_value

    def weights_init(m):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform_(m.weight)

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
