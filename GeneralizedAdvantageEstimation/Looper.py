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

    def __init__(self, env="LunarLanderContinuous-v2", gamma=0.99):
        self.policy = Policy(env_id=env)
        self.env = gym.make(env)
        self.runs = Runs(gamma=gamma)

        self.plotter = VisdomPlotter(env_name=env)

        self.device_cpu = torch.device("cpu")
        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

    def loop(self, epochs=1,
             show_every=50,
             show_for=10,
             sample_count=64,
             critic_lr=0.001,
             critic_batch=128,
             critic_iterations=64,
             critic_max_loss=0.1,
             actor_lr=0.001,
             actor_batch=128,
             actor_iterations=1,
             actor_lambda=0):

        for e in trange(epochs):

            if e % show_every == 0 and e > 0:
                self.policy.demonstrate(ep_count=show_for)

            reward = self.generate_samples(num_ep=sample_count)
            self.plotter.plot_line('reward per episode', 'reward', 'avg reward when generating samples', e, reward)

            loss, last_avg_loss = self.estimate_return(lr=critic_lr,
                                                       batch_size=critic_batch,
                                                       iterations=critic_iterations,
                                                       critic_max_loss=critic_max_loss)
            self.plotter.plot_line('loss for critic fit', 'loss', 'avg loss per batch', e, loss)
            self.plotter.plot_line('loss for critic fit', 'last_loss', 'avg loss per batch', e, last_avg_loss)
            self.improve_policy(lr=actor_lr, batch_size=actor_batch, iterations=actor_iterations, _lambda_=actor_lambda)

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
                    reward += r
                    self.runs.add_next(state=ob, action=action, reward=r, next_state=ob_, done=done)
                    ob = ob_
                    if render:
                        self.env.render()
        reward /= num_ep
        return reward

    def estimate_return(self, lr=0.001, batch_size=128, iterations=1024, critic_max_loss=0.1, ):
        self.runs.compute_rewards()
        data_collected = list(self.runs.state_target.items())
        data_loader = CriticLoader(data_collected=data_collected)
        loss, last_avg_loss = self.policy.fit_critic(data_loader=data_loader,
                                                     lr=lr,
                                                     batch_size=batch_size,
                                                     iterations=iterations,
                                                     critic_max_loss=critic_max_loss)
        return loss, last_avg_loss

    def improve_policy(self, lr=0.001, batch_size=128, iterations=1, _lambda_=0):
        with torch.no_grad():
            self.runs.compute_advantage(v=self.policy.critic, _lambda_=_lambda_)
        data_collected = list(self.runs.advantage_sa_mean.items())
        data_loader = ActorLoader(data_collected=data_collected)
        self.policy.improve_actor(data_loader=data_loader, lr=lr, batch_size=batch_size, iterations=iterations)


if __name__ == '__main__':
    looper = Looper(env="LunarLanderContinuous-v2", gamma=0.99)
    # looper.policy.demonstrate(ep_count=1)
    looper.loop(epochs=1000,
                show_every=1000,
                show_for=5,
                sample_count=64,
                critic_lr=0.001,
                critic_batch=64,
                critic_iterations=128,
                critic_max_loss=1,
                actor_lr=0.001,
                actor_batch=64,
                actor_iterations=1,
                actor_lambda=0)
    looper.policy.save_policy(save_name="attempt1")
    looper.policy.demonstrate(ep_count=10)
