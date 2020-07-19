from UnitStep import UnitStep


class Trajectory:

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.trajectory = []
        # self.states = []
        # self.actions = []
        # self.rewards = []
        # self.next_states = []
        # self.dones = []
        # self.rtgs = []

    def __len__(self):
        return len(self.trajectory)

    def copy_to_current(self, instance):
        self.gamma = instance.gamma
        self.trajectory = instance.trajectory
        # self.states = instance.states
        # self.actions = instance.actions
        # self.rewards = instance.rewards
        # self.next_states = instance.next_states
        # self.dones = instance.dones
        # self.rtgs = instance.rtgs

    def reset(self):
        self.trajectory = []

    def add_step(self, state, action, reward, next_state, done=False):

        step = UnitStep(state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                        rtg=0)

        self.trajectory.append(step)
        # self.states.append(step.state)
        # self.actions.append(step.action)
        # self.rewards.append(step.reward)
        # self.next_states.append(step.next_state)
        # self.dones.append(step.done)
        # self.rtgs.append(step.rtg)

    def add_step_directly(self, step):
        # step is of type UnitStep
        self.trajectory.append(step)
        # self.states.append(step.state)
        # self.actions.append(step.action)
        # self.rewards.append(step.reward)
        # self.next_states.append(step.next_state)
        # self.dones.append(step.done)
        # self.rtgs.append(step.rtg)

    def compute_rtg(self):
        # call this after trajectory has ended.
        last_step = self.trajectory[-1]
        if not last_step.done:
            raise Exception("Trajectory not complete!")

        self.trajectory[-1].rtg = self.trajectory[-1].reward
        # self.rtgs[-1] = self.rewards[-1]

        n = self.__len__() - 2
        if n < 0:
            return

        for i in range(n, 0, -1):
            # self.rtgs[i] += (self.gamma*self.rtgs[i+1])
            self.trajectory[i].rtg += (self.gamma * self.trajectory[i + 1].rtg)


if __name__ == '__main__':
    print("Write code to test trajectory here")
