import random


class UnitStep:

    def __init__(self,
                 state=None,
                 action=None,
                 reward=None,
                 next_state=None,
                 done=None,
                 rtg=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.rtg = rtg

    def set_random_value(self, state_size=8, action_size=2):
        # use this function for quick tests only
        self.state = [random.random() for iter in range(state_size)]
        self.next_state = [random.random() for iter in range(state_size)]
        self.action = [random.random() for iter in range(action_size)]
        self.reward = random.randint(0, 100)
        self.rtg = random.randint(0, 100)
        self.done = False


if __name__ == '__main__':
    import random

    a = random.randint(0, 100)

    print(a)
