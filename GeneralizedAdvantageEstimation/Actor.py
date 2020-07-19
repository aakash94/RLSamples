import torch
import torch.nn as nn
import torch.functional as F


class Actor(nn.Module):

    # For continuous actions
    # returning mean and standard deviation

    def __init__(self, n_ip, n_op, move_to_gpu=True):
        super(Actor, self).__init__()

        # Will sharing parameters here help? IDK.

        self.m_fc1 = nn.Linear(n_ip, n_ip * 16)
        self.m_fc2 = nn.Linear(n_ip * 16, n_ip * 16)
        self.m_fc3 = nn.Linear(n_ip * 16, n_op)

        self.s_fc1 = nn.Linear(n_ip, n_ip * 16)
        self.s_fc2 = nn.Linear(n_ip * 16, n_ip * 16)
        self.s_fc3 = nn.Linear(n_ip * 16, n_op)

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        if move_to_gpu:
            self.to_device()

    def forward(self, x):

        # x = x.to(self.device)

        # using tanh here for mean
        # most env take value in that range

        t_mean = torch.tanh(self.m_fc1(x))
        t_mean = torch.tanh(self.m_fc2(t_mean))
        t_mean = torch.tanh(self.m_fc3(t_mean))

        t_std = torch.tanh(self.s_fc1(x))
        t_std = torch.relu(self.s_fc2(t_std))
        t_std = torch.relu(self.s_fc3(t_std))
        t_std = torch.clamp(t_std, min=0.001)

        # t_mean=t_mean.to("cpu")
        # t_std = t_std.to("cpu")
        return (t_mean, t_std)

    def save_model(self, save_name):
        self.to_cpu()
        save_path = "saves/" + save_name + ".chkpt"
        torch.save(self.state_dict(), save_path)
        self.to_device()

    def load_model(self, save_name):
        self.to_cpu()
        save_path = "saves/" + save_name + ".chkpt"
        self.load_state_dict(torch.load(save_path))
        self.to_device()

    def to_device(self):
        if self.use_gpu:
            self.cuda()

    def to_cpu(self):
        self.cpu()
