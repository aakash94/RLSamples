from Actor import Actor
from Critic import Critic
from tqdm import trange
from Runs import Runs
from Policy import Policy
from Loader import CriticLoader
from Loader import ActorLoader
import gym
import torch
import numpy as np


class Looper:

    def __init__(self, env="LunarLanderContinuous-v2"):
        self.policy = Policy(env_id=env)
        self.env = gym.make(env)
        self.runs = Runs()

        self.device_cpu = torch.device("cpu")
        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

    def loop(self, epochs):

        render = False
        for e in trange(epochs):
            if e % 10 == 0:
                render = True
            else:
                render = False

            self.generate_samples(render=render)
            self.estimate_return()
            self.improve_policy()

    def generate_samples(self, num_ep=1, render=False):
        # Add stuff to trajectories
        self.runs.reset()

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
                    self.runs.add_next(state=ob, action=action, reward=r, next_state=ob_, done=done)
                    if render:
                        self.env.render()

    def estimate_return(self, lr=0.001, batch_size=128, iterations=1024):
        self.runs.compute_rewards()
        data_collected = list(self.runs.state_target.items())
        data_loader = CriticLoader(data_collected=data_collected)
        self.policy.fit_critic(data_loader=data_loader, lr=lr, batch_size=batch_size, iterations=iterations)

    def improve_policy(self, lr=0.001, batch_size=128, iterations=1):
        with torch.no_grad():
            self.runs.compute_advantage()
        data_collected = list(self.runs.advantage_sa_mean.items())
        data_loader = ActorLoader(data_collected=data_collected)
        self.policy.improve_actor(data_loader=data_loader, lr=lr, batch_size=batch_size, iterations=iterations)


if __name__ == '__main__':
    # TODO: Add logging
    pass