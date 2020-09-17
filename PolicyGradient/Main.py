from Looper import Looper


def main():
    '''
    READ ME:

    Every python file(apart from this) is a class

    structured in the 3 part form as told in cs285
    * collect data, *estimate return, *improve policy
    all of the above are called in a loop in looper

    Episode makes Runs. Each episode is a pandas dataframe.
    Runs is for everything related to a set of roll outs.
    Everything specific to a set of rollouts happen here.

    Policy is made out of an actor and a critic
    responsible for training actor and critic
    also samples action and calculates action probability

    Loader had dataloaders for training actor and critic

    VisdomPlotter is used to plot Loss and Rewards in Visdom

    In runs, to avoid passing states one at a time through the critic network,
    advantage is calculated in batches and stored in a dict.
    The dict is later used in the the dataloader for improving actor.
    '''
    
    looper = Looper(env="Pendulum-v0", gamma=0.99)
    looper.loop(epochs=100,
                show_every=10000,
                show_for=5,
                sample_count=1024,
                lr=5e-3,
                batch_size=4096)

    looper.policy.save_policy(save_name="PV0-a1")
    looper.policy.demonstrate(ep_count=10)


if __name__ == '__main__':
    main()
