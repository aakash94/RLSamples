import numpy as np


class UnitStep:

    def __init__(self,
                 state=np.zeros(1),
                 action=np.zeros(1),
                 reward=None,
                 next_state=np.zeros(1),
                 done=None,
                 rtg=None):
        self.state = state.astype('float32')
        self.action = action.astype('float32')
        self.next_state = next_state.astype('float32')
        self.reward = reward
        self.done = done
        self.rtg = rtg

    def set_random_value(self, state_size=3, action_size=2):
        self.state = np.random.rand(state_size)
        self.next_state = np.random.rand(state_size)
        self.action = np.random.rand(action_size)
        self.reward = random.randint(0, 100)
        self.rtg = 0  # random.randint(0, 100)
        self.done = False

    def to_string(self, print_it=False):
        s = str([round(num, 1) for num in self.state]) + "\t" + \
            str([round(num, 1) for num in self.action]) + "\t" + \
            str(self.reward) + "\t" + \
            str([round(num, 1) for num in self.next_state]) + "\t" + \
            str(self.done) + "\t" + \
            str(self.rtg)
        if print_it:
            print(s)

        return s


if __name__ == '__main__':
    import random

    a = UnitStep()
    a.set_random_value()

    a.to_string(print_it=True)
