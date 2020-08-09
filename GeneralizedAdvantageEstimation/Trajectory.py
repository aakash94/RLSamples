from UnitStep import UnitStep


class Trajectory:

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.trajectory = []

    def __len__(self):
        return len(self.trajectory)

    def copy_to_current(self, instance):
        self.gamma = instance.gamma
        self.trajectory = instance.trajectory

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

    def add_step_directly(self, step):
        # step is of type UnitStep
        self.trajectory.append(step)

    def compute_rtg(self):
        # call this after trajectory has ended.
        last_step = self.trajectory[-1]
        if not last_step.done:
            raise Exception("Trajectory not complete!")

        self.trajectory[-1].rtg = self.trajectory[-1].reward

        n = self.__len__() - 2

        if n < 0:
            return

        for i in range(n, -1, -1):
            self.trajectory[i].rtg = self.trajectory[i].reward + (self.gamma * self.trajectory[i + 1].rtg)

    def to_string(self):
        print("gamma\n", self.gamma)
        for us in self.trajectory:
            us.to_string(print_it=True)



if __name__ == '__main__':
    t = Trajectory(gamma=1)
    time_steps = 10
    for i in range(time_steps):
        us = UnitStep()
        us.set_random_value()
        if i == time_steps-1:
            us.done = True
        t.add_step(state=us.state, action=us.action, reward=us.reward, next_state=us.next_state, done=us.done)

    t.to_string()
    t.compute_rtg()
    t.to_string()
