from Actor import Actor
from Runs import Runs
import torch.utils.data.dataloader as dataloader
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from VisdomPlotter import VisdomPlotter


class Policy:

    def save_policy(self, save_name):
        self.actor.save_model(save_name=save_name)

    def load_policy(self, load_name):
        self.actor.load_model(load_name=load_name)

    def __init__(self, env_id="LunarLanderContinuous-v2"):
        self.envid = env_id
        env = gym.make(env_id)
        state_size = np.prod(list(env.observation_space.shape))
        action_size = np.prod(list(env.action_space.shape))

        # the max and min here are shorcut
        # because all spaces have same range here
        # ideally when clamping, different dimension should support different ranges
        self.low_action = env.action_space.low.max()
        self.high_action = env.action_space.high.min()

        self.actor = Actor(n_ip=state_size, n_op=action_size)

        self.actor.apply(self.weights_init)

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

    def weights_init(self, m):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def sample_action(self, observations):
        # observations = [ [], [], [], [] ]
        ob = torch.Tensor(observations).to(self.device)
        m, s = self.actor(ob)
        chnk = len(m[0])  # the size of action space here
        m = m.cpu().flatten().float()
        s = s.cpu().flatten().float()
        samples = torch.normal(mean=m, std=s)
        samples = torch.clamp(samples, min=self.low_action, max=self.high_action)
        sampled_action = samples.reshape(-1, chnk).detach().numpy()
        return sampled_action

    def get_log_prob(self, mean, standard_deviation, actions):

        m = mean
        s = standard_deviation
        log_prob = torch.distributions.Normal(loc=m, scale=s).log_prob(actions)  # .sum(-1)
        return log_prob

    def improve_actor(self, data_loader, lr=0.001, batch_size=128, iterations=1):
        optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        loader = dataloader.DataLoader(data_loader, batch_size=batch_size, shuffle=True)
        for e in range(iterations):
            for states, actions, values in loader:
                optimizer.zero_grad()
                m, s = self.actor(states)
                lp = self.get_log_prob(mean=m, standard_deviation=s, actions=actions)
                loss = -(lp * values).sum()
                loss.backward()
                optimizer.step()

    def demonstrate(self, ep_count=1):
        env = gym.make(self.envid)
        with torch.no_grad():
            for e in range(ep_count):
                done = False
                ob = env.reset()
                while not done:
                    observation = ob[None]
                    action = self.sample_action(observations=observation)
                    action = action[0]
                    ob_, r, done, _ = env.step(action)
                    env.render()
        env.close()


if __name__ == '__main__':
    p = Policy()
    m = [100, 200, 300]
    s = [0.3, 0.3, 0.9]
    a = [100, 200, 300]
    v = [1000, 100, 10]
    m = torch.tensor(m).float()
    s = torch.tensor(s).float()
    a = torch.tensor(a).float()
    v = torch.tensor(v).float()
    lp = p.get_log_prob(mean=m, standard_deviation=s, actions=a)
    print(lp)
    nlp = -(lp * v)  # .sum()
    print(nlp)