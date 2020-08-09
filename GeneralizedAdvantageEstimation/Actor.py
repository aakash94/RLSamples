import torch
import torch.nn as nn


class Actor(nn.Module):

    # For continuous actions
    # returning mean and standard deviation

    def __init__(self, n_ip, n_op, min_std_start=0.5, move_to_gpu=True, std_min=1e-3, std_max=2, action_scale=2):
        super(Actor, self).__init__()

        self.std_min = std_min
        self.std_max = std_max
        self.action_scale = action_scale

        self.fc10 = nn.Linear(n_ip, n_ip * 16)
        self.fc20 = nn.Linear(n_ip * 16, n_ip * 32)
        self.fc_mu = nn.Linear(n_ip * 32, n_op)
        self.fc_std = nn.Linear(n_ip * 32, n_op)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()


        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        if move_to_gpu:
            self.to_device()

    def forward(self, x):

        x = self.relu(self.fc10(x))
        x = self.relu(self.fc20(x))
        mu = self.action_scale * self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x))

        #std = self.clip_std_val(std=std, min=self.std_min, max=self.std_max)
        std = std.clamp(min=self.std_min, max=self.std_max)

        return (mu, std)

    def clip_log_val(self, min=-5, max=0):
        if hasattr(self.t_logstd, 'data'):
            w = self.t_logstd.data
            w = w.clamp(max=max)
            self.t_logstd.data = w
        else:
            print("WTF")

    def clip_std_val(self, std, min=1e-3,  max=2):
        if hasattr(self.std, 'data'):
            w = std.data
            w = w.clamp(min=min, max=max)
            std.data = w
            return  std
        else:
            print("WTF")

    def get_std_values(self):
        # v = self.t_logstd.exp()
        v = self.t_std
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
    a = nn.Parameter(torch.randn((2)) + 2)  # +2

    print(a)
