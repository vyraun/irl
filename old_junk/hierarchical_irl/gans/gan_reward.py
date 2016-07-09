import tensorflow as tf

class GAN:
    def __init__(self, policy):
        self.generator = Generator(policy)
        self.discriminator = Discriminator(policy)


class Generator:
    # essentially a policy.
    def __init__(self, policy):
        self.policy = policy

    def train(self):
        self.policy.train()

    def rollout(self):
        self.policy.rollout()


class Discriminator:
    # Train a discriminator used to distribute reward.
    def __init__(self, policy, master_policy_samples):
        self.policy = policy
        self.rollouts = []
        self.master_policy_samples = master_policy_samples

    def train(self, num_rollouts=10):
        self.rollouts = []
        for iter_step in range(0, num_rollouts):
            self.rollouts.append(self.policy.rollout())

    def build_disc_mlp_net(self, nn_input_shape):
        net_input = tf.placeholder("float", [None, nn_input_shape], name='nn_input')


    def build_disc_conv_net(self, nn_input_shape):
        pass


def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)




