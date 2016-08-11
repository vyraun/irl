from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.npo import NPO
#from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
#from sandbox.rocky.tf.policies.uniform_control_policy import UniformControlPolicy
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv


def gen_expert_data():
    stub(globals())

    env = TfEnv(normalize(GymEnv('Impr2-v0')))
    #env = normalize(GymEnv('Imreacher-v0'))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
    #    # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    #policy = UniformControlPolicy(env_spec=env.spec)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=100,
        discount=0.99,
        step_size=0.01,
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=1,
        seed=1,
    )


gen_expert_data()
