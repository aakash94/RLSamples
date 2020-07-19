from Actor import Actor
from Critic import Critic
from Runs import Runs
import torch.utils.data.dataloader as dataloader
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Policy:

    def __init__(self, env_id="LunarLanderContinuous-v2"):
        env = gym.make(env_id)
        state_size = np.prod(list(env.observation_space.shape))
        action_size = np.prod(list(env.action_space.shape))

        self.actor = Actor(n_ip=state_size, n_op=action_size).apply(self.weights_init)
        self.critic = Critic(n_ip=state_size).apply(self.weights_init)

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

    def weights_init(m):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform_(m.weight)

    def fit_critic(self, data_loader, lr=0.001, batch_size=128, iterations=1024):
        loader = dataloader.DataLoader(data_loader, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()  # for controlled gradients
        for i in range(iterations):
            sum_loss = 0.0
            count = 0
            for x, y in loader:

                # if self.use_gpu:
                #     x = x.float().cuda()
                #     y = y.float().cuda()

                optimizer.zero_grad()
                output = self.critic(x)
                loss = criterion(output, y)
                sum_loss += loss.item()
                count += 1
                loss.backward()
                optimizer.step()

            total_loss += (sum_loss / count)

        average_loss = total_loss / iterations

    def sample_action(self, observations):
        # observations = [ [], [], [], [] ]
        ob = torch.Tensor(observations).to(self.device)
        m, s = self.actor(ob)
        chnk = len(m[0])  # the size of action space here
        m = m.cpu().flatten().float()
        s = s.cpu().flatten().float()
        samples = torch.normal(mean=m, std=s)
        sampled_action = samples.reshape(-1, chnk).numpy()
        return sampled_action

    def get_log_prob(self, mean, standard_deviation, actions):
        # TODO: Adjust return value
        m = mean
        s = standard_deviation
        log_prob = torch.distributions.Normal(loc=m, scale=s).log_prob(actions).sum(-1)
        return log_prob

    def improve_actor(self, data_loader, lr=0.001, batch_size=128, iterations=1):
        optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        loader = dataloader.DataLoader(data_loader, batch_size=batch_size, shuffle=True)
        for e in range(iterations):
            sum_loss = 0.0
            count = 0
            for states, actions, values in loader:
                optimizer.zero_grad()
                m, s = self.actor(states)
                # TODO: Complete This


    def save_policy(self):
        # TODO : Fill save_policy
        pass

    def load_policy(self):
        # TODO : Fill load_policy
        pass

    def demonstrate(self):
        # TODO : Fill demonstrate
        pass
