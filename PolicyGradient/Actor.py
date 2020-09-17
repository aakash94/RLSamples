import torch
import torch.nn as nn


class Actor(nn.Module):

    # For continuous actions
    # returning mean and standard deviation

    def __init__(self, n_ip, n_op, move_to_gpu=True):
        super(Actor, self).__init__()
        size = 64
        activation = nn.Tanh()
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(n_ip, size))
        self.mlp.append(activation)
        self.mlp.append(nn.Linear(size, size))
        self.mlp.append(activation)
        self.mlp.append(nn.Linear(size, n_op))  # output value

        self.logstd = nn.Parameter(torch.zeros(n_op))

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        if move_to_gpu:
            self.to_device()

    def forward(self, x):

        for layer in self.mlp:
            x = layer(x)

        return (x, self.logstd.exp())

    def get_std_values(self):
        with torch.no_grad:
            v = self.logstd.exp().clone()

        v = v.detach().cpu().numpy()
        return v

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


if __name__ == '__main__':
    pass