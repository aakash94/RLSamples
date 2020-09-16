import gym
import torch
from tqdm import trange

from Loader import ActorLoader
from Loader import CriticLoader
from Policy import Policy
from Runs import Runs
from VisdomPlotter import VisdomPlotter


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
             sample_count=1,
             lr=5e-3,
             batch_size=4096):

        for e in trange(epochs):

            if e % show_every == 0 and e > 0:
                #self.policy.demonstrate(ep_count=show_for)
                pass

            reward = self.generate_samples(num_ep=sample_count)
            self.plotter.plot_line('reward per episode', 'reward', 'avg reward when generating samples', e, reward)
            loss = self.estimate_return(lr=lr, batch_size=batch_size)
            self.plotter.plot_line('loss per batch', 'loss', 'avg loss when training critic', e, loss)
            self.improve_policy(lr=lr, batch_size=batch_size)

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

    def estimate_return(self, lr=5e-3, batch_size=4096):
        self.runs.compute_rewards()
        vs = self.runs.get_normalized_rtg()
        dataloader = CriticLoader(dframe=vs)
        loss = self.policy.improve_critic(data_loader=dataloader, lr=lr, batch_size=batch_size)
        self.runs.compute_baseline_dict(critic=self.policy.critic, batch_size=batch_size)
        return loss

    def improve_policy(self, lr=5e-3, batch_size=4096):

        data_loader = ActorLoader(dframe=self.runs.all_runs, baseline_dict=self.runs.baseline)
        self.policy.improve_actor(data_loader=data_loader, lr=lr, batch_size=batch_size)


if __name__ == '__main__':
    looper = Looper(env="Pendulum-v0", gamma=0.99)
    #looper.policy.demonstrate(ep_count=10)
    looper.loop(epochs=100,
                show_every=10000,
                show_for=5,
                sample_count=4096,
                lr=5e-3,
                batch_size=4096)

    # looper.loop(epochs=2,
    #             show_every=10000,
    #             show_for=5,
    #             sample_count=4,
    #             lr=5e-3,
    #             batch_size=512)


    looper.policy.save_policy(save_name="PV0-a1")
    looper.policy.demonstrate(ep_count=10)
