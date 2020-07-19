from Actor import Actor
from Critic import Critic
from tqdm import trange
from Runs import Runs
from Policy import Policy
from Loader import CriticLoader
from Loader import ActorLoader
from VisdomPlotter import VisdomPlotter
import gym
import torch
import numpy as np


class Looper:

    def __init__(self, env="LunarLanderContinuous-v2"):
        self.policy = Policy(env_id=env)
        self.env = gym.make(env)
        self.runs = Runs()

        self.plotter = VisdomPlotter(env_name=env)

        self.device_cpu = torch.device("cpu")
        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

    def loop(self, epochs, show_every=100):

        for e in trange(epochs):

            if e % show_every == 0:
                self.policy.demonstrate(ep_count=10)

            reward = self.generate_samples()
            self.plotter.plot_line('reward per episode', 'reward', 'avg reward when generating samples', e, reward)
            loss = self.estimate_return()
            self.plotter.plot_line('loss for critic fit', 'loss', 'avg loss per batch', e, loss)
            self.improve_policy()

    def generate_samples(self, num_ep=1, render=False):
        # Add stuff to trajectories
        self.runs.reset()
        reward = 0

        with torch.no_grad():
            for e in range(num_ep):
                # the eth episode in this run
                done = False
                ob = self.env.reset()
                while not done:
                    observation = ob[None]
                    action = self.policy.sample_action(observations=observation)
                    action = action[0]
                    ob_, r, done, _ = self.env.step(action)
                    reward+=r
                    self.runs.add_next(state=ob, action=action, reward=r, next_state=ob_, done=done)
                    if render:
                        self.env.render()
        reward/=num_ep
        return reward

    def estimate_return(self, lr=0.001, batch_size=128, iterations=1024):
        self.runs.compute_rewards()
        data_collected = list(self.runs.state_target.items())
        data_loader = CriticLoader(data_collected=data_collected)
        loss = self.policy.fit_critic(data_loader=data_loader, lr=lr, batch_size=batch_size, iterations=iterations)
        return loss

    def improve_policy(self, lr=0.001, batch_size=128, iterations=1):
        with torch.no_grad():
            self.runs.compute_advantage()
        data_collected = list(self.runs.advantage_sa_mean.items())
        data_loader = ActorLoader(data_collected=data_collected)
        self.policy.improve_actor(data_loader=data_loader, lr=lr, batch_size=batch_size, iterations=iterations)


if __name__ == '__main__':
    looper = Looper()