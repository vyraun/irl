#from irl.generators import TRPOGenerator
from point_env import PointEnv

from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite


def gen_expert_data():
    stub(globals())

    env = TfEnv(normalize(PointEnv()))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=6000,
        max_path_length=100,
        n_itr=80,
        discount=0.99,
        step_size=0.01,
    )
    #algo.train()
    run_experiment_lite(
        algo.train(),
        n_parallel=1,
        seed=1,
    )


gen_expert_data()
