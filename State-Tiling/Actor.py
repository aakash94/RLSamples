import torch
import torch.nn as nn


class Actor(nn.Module):

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

    def __init__(self,
                 n_ip,
                 n_op,
                 n_hidden=64,
                 hidden_layers=2,
                 move_to_gpu=True,
                 activation=nn.Tanh(),
                 action_scale=2):
        super(Actor, self).__init__()

        # range is -2 to 2
        self.action_scale = action_scale

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(n_ip, n_hidden))  # first hidden layer
        self.mlp.append(activation)

        for h in range(hidden_layers - 1):  # additional hidden layers
            self.mlp.append(nn.Linear(n_hidden, n_hidden))
            self.mlp.append(activation)

        self.mlp.append(nn.Linear(n_hidden, n_op))  # output layer, no activation function

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

    def clip_log_val(self, min=-5, max=0):
        if hasattr(self.t_logstd, 'data'):
            w = self.t_logstd.data
            w = w.clamp(max=max)
            self.t_logstd.data = w
        else:
            print("WTF")

    def clip_std_val(self, std, min=1e-3, max=2):
        if hasattr(self.std, 'data'):
            w = std.data
            w = w.clamp(min=min, max=max)
            std.data = w
            return std
        else:
            print("WTF")

    def get_std_values(self):
        v = self.logstd.exp()
        v = v.detach().cpu().numpy()
        return v


if __name__ == '__main__':
    a = nn.Parameter(torch.randn((2)) + 2)  # +2

    print(a)
