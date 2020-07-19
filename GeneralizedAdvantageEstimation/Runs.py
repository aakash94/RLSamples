from Critic import Critic
from UnitStep import UnitStep
from collections import defaultdict
from Trajectory import Trajectory
from Loader import StatesLoader
import torch.utils.data.dataloader as dataloader
import numpy as np


class Runs:

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.trajectories = []

        # used for v pi
        self.state_reward = defaultdict(list)
        self.state_target = {}

        # used for advantage
        self.reward_sa = defaultdict(list)
        self.reward_sa_mean = {}

        # used for advantage
        self.advantage_sa = defaultdict(list)
        self.advantage_sa_mean = {}

        # for whatever trajectory is added first here
        trajectory = Trajectory(gamma=self.gamma)
        self.trajectories.append(trajectory)

    def add_next(self, state, action, reward, next_state, done):

        self.trajectories[-1].add_step(state=state,
                                       action=action,
                                       reward=reward,
                                       next_state=next_state,
                                       done=done)
        if done:
            self.trajectories[-1].compute_rtg()
            next_trajectory = Trajectory()
            self.trajectories.append(next_trajectory)

    def add_next_step(self, step):
        self.trajectories[-1].add_step_directly(step=step)
        if step.done:
            self.trajectories[-1].compute_rtg()
            next_trajectory = Trajectory()
            self.trajectories.append(next_trajectory)

    def compute_rewards(self):
        # call this only after all runs are executed
        # should use running average here, not doing so for sake of simplicity
        for trajectory in self.trajectories:

            for step in trajectory.trajectory:
                # convert to tuple because they will be keys for dictionary
                state = tuple(step.state)
                action = tuple(step.action)
                rtg = tuple(step.reward)
                self.state_reward[(state)].append(rtg)
                self.reward_sa[(state, action)].append(rtg)

            for key, value in self.reward_sa:
                # key is state action pair
                # value is list of rewards.
                state = key[0]
                action = key[1]
                average_sa = sum(value) / len(value)
                self.reward_sa_mean[key] = average_sa

                if state not in self.state_target:
                    # just to avoid extra computation
                    v = self.state_reward[state]
                    avg_v = sum(v) / len(v)
                    self.state_target[state] = avg_v

    def compute_advantage(self, v, _lambda_=0):

        alpha = self.gamma * _lambda_
        for t in self.trajectories:
            trajectory = t.trajectory
            # trajectory = [ UnitStep, UnitStep, UnitStep, UnitStep, ]
            data_loader = StatesLoader(trajectory=trajectory)
            v_s = self.get_state_values(data_loader=data_loader, crtc=v, batch_size=128)

            len_trajectory = len(trajectory)
            d_t = np.zeros(len_trajectory)

            last_step = trajectory[-1]
            last_state = tuple(last_step.state)
            last_action = tuple(last_step.action)
            d_t[-1] = self.reward_sa_mean[(last_state, last_action)]

            n = len_trajectory - 2
            if n < 0:
                return

            for i in range(n, 0, -1):
                step = trajectory[i]
                state = tuple(step.state)
                action = tuple(step.action)
                next_state = tuple(step.next_state)
                v_now = v_s[state]
                v_next = v_s[next_state]
                d_t[i] = self.reward_sa_mean[(state, action)] + (self.gamma * v_next) - (v_now)

                if _lambda_ == 0:
                    # slight optimization to avoid unnecessary sums
                    self.advantage_sa[(state, action)].append(d_t[i])
                else:
                    self.advantage_sa[(state, action)].append(d_t.sum())
                    d_t = d_t*alpha

        for key, value in self.advantage_sa:
            # key is state action pair
            # value is list of rewards.
            average_sa = sum(value) / len(value)
            self.advantage_sa_mean[key] = average_sa




    def get_state_values(self, data_loader, crtc, batch_size=32):
        v_s = {}
        loader = dataloader.DataLoader(data_loader, batch_size=batch_size, shuffle=True)
        for states in loader:
            with torch.no_grad():
                values = crtc(states)
                values = values.cpu().flatten().numpy()

            states = states.cpu().numpy()
            states = [tuple(state) for state in states]
            d = dict(zip(states, values))
            v_s.update(d)
        return v_s


if __name__ == '__main__':
    print("Use to store trajectories and calculate values")

    import torch

    t = []

    for i in range(30):
        s = UnitStep()
        s.set_random_value(state_size=3, action_size=2)
        t.append(s)

    print("t\n", t)

    c = Critic(n_ip=3)

    data_loader = StatesLoader(trajectory=t)
    loader = dataloader.DataLoader(data_loader, batch_size=10, shuffle=True)
    d_final = {}
    for x in loader:
        with torch.no_grad():
            v_s = c(x)
            v_s = v_s.cpu().flatten().numpy()

        x = x.cpu().numpy()

        x = [tuple(a) for a in x]

        # print("\n\nX\n", x)
        # print("\nstate\n", state)
        # print("\nv_s\n", v_s)

        d = dict(zip(x, v_s))
        # print("\nd\n", d)

        d_final.update(d)

    print(len(d_final))
    print(d_final)
