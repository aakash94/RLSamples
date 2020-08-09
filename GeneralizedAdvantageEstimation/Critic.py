import torch
import torch.nn as nn


class Critic(nn.Module):

    # For continuous actions
    # returning mean and standard deviation

    def __init__(self, n_ip, drop_p=0.0, move_to_gpu=True):
        super(Critic, self).__init__()
        self.fc10 = nn.Linear(n_ip, n_ip * 16)
        self.fc20 = nn.Linear(n_ip * 16, n_ip * 32)
        #self.fc30 = nn.Linear(n_ip * 8, n_ip * 8)
        self.fc40 = nn.Linear(n_ip * 32, 1)
        self.drop_prob = drop_p

        if self.drop_prob > 0:
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
        x = torch.tanh(self.fc10(x))

        if self.drop_prob > 0:
            x = self.dropout(x)

        x = torch.tanh(self.fc20(x))

        if self.drop_prob > 0:
            x = self.dropout(x)

        # x = torch.tanh(self.fc30(x))
        #
        # if self.drop_prob > 0:
        #     x = self.dropout(x)

        t_value = self.fc40(x)

        return t_value

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
