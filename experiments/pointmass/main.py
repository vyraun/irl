from irl.generators import TRPOGenerator
from irl.discriminators import DomainConfusionDiscriminator
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize


def main(steps=10, gen_lr=0.1, disc_lr=0.1, env=None):
    gen = TRPOGenerator(env=env)
    disc = DomainConfusionDiscriminator()


if __name__ == '__main__':
    #env = normalize(GymEnv("Pendulum-v0"))
    #main(env=env)
    pass