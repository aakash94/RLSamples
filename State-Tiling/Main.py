from tqdm import trange
from Runs import Runs
from Policy import Policy
from Loader import Loader
from VisdomPlotter import VisdomPlotter
import gym
import torch


class Looper:

    def __init__(self, env="LunarLanderContinuous-v2", gamma=0.99, round_decimal=3):
        self.policy = Policy(env_id=env)
        self.env = gym.make(env)
        self.runs = Runs(gamma=gamma)
        self.round_decimal = round_decimal

        self.plotter = VisdomPlotter(env_name=env)

        self.device_cpu = torch.device("cpu")
        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

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
                    self.runs.add_step(state=ob, action=action, reward=r, next_state=ob_, done=done)
                    ob = ob_
                    if render:
                        self.env.render()
        reward /= num_ep
        return reward

    def estimate_return(self):
        self.runs.compute_rewards(round_decimals=self.round_decimal)

    def improve_policy(self, lr=0.001, batch_size=128, iterations=1):
        data_loader = Loader(vs_round=self.runs.v_s_rounded, qsa=self.runs.qsa, round_decimals=self.round_decimal)
        self.policy.improve_actor(data_loader=data_loader, lr=lr, batch_size=batch_size, iterations=iterations)

    def loop(self,
             epochs=1,
             show_every=1,
             show_for=1,
             sample_count=1,
             lr=0.001,
             batch=1,
             iterations=1):

        for e in trange(epochs):

            if e % show_every == 0 and e > 0:
                self.policy.demonstrate(ep_count=show_for)

            temp_std = self.policy.actor.get_std_values()
            self.plotter.plot_line('actor std', 'dimension 0', 'standard deviation of action dimension', e, temp_std[0])

            reward = self.generate_samples(num_ep=sample_count)
            self.plotter.plot_line('reward per episode', 'reward', 'avg reward when generating samples', e, reward)

            self.estimate_return()

            self.improve_policy(lr=lr, batch_size=batch, iterations=iterations)


def main():
    print("Hello World")
    looper = Looper(env="Pendulum-v0", gamma=0.99, round_decimal=2)
    looper.loop(epochs=1000,
                show_every=10000,
                show_for=5,
                sample_count=1024,
                lr=0.001,
                batch=128)
    looper.policy.save_policy(save_name="ST0-a1")
    looper.policy.demonstrate(ep_count=10)


if __name__ == '__main__':
    main()
