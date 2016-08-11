from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite


class TRPOGenerator:
    def __init__(self, env=None):
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 8 hidden units.
            hidden_sizes=(8, 8)
        )
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        self.im_trpo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=env.horizon,
            n_itr=50,
            discount=0.99,
            step_size=0.01,
        )

    def train(self):
        self.im_trpo.train()

    def update_trpo_reward(self):
        pass

    def __call__(self):
        return self.im_trpo.sampler.obtain_samples()






