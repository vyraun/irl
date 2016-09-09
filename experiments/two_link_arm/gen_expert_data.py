from sandbox.rocky.tf.algos.trpo import TRPO
#from sandbox.rocky.tf.algos.npo import NPO
#from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv

import os
import math


def gen_expert_data(save_dir, expert=True, camera_angle=0, batch_size=6000, burn_in=10, save_steps=10):
    #stub(globals())

    env = TfEnv(normalize(GymEnv('Imreacher-v0')))
    #env = normalize(GymEnv('Imreacher-v0'))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
    #    # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    env._wrapped_env._wrapped_env.env.save_dir = save_dir
    if expert is True:
        env._wrapped_env._wrapped_env.env.reward_function = env._wrapped_env._wrapped_env.env.standard_reward
    else:
        env._wrapped_env._wrapped_env.env.reward_function = env._wrapped_env._wrapped_env.env.zero_reward

    env._wrapped_env._wrapped_env.env.generation_index = camera_angle
    env._wrapped_env._wrapped_env.env.reset_viewer = True
    env._wrapped_env._wrapped_env.env.save_steps = save_steps
    env._wrapped_env._wrapped_env.env.burn_in_steps = burn_in
    total_iters = int(math.ceil((1.0*burn_in + save_steps) / batch_size) + 3)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=200,
        n_itr=total_iters,
        discount=0.99,
        step_size=0.01,
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=1,
        seed=1,
    )


if __name__ == '__main__':
    master_path = os.getcwd()
    expert_one_path = master_path + '/domain_one_success'
    expert_two_path = master_path + '/domain_two_success'
    bad_one_path = master_path + '/domain_one_failure'
    bad_two_path = master_path + '/domain_two_failure'

    save_steps = 10000
    batch_size = 6000
    burn_in_expert = 15*batch_size
    burn_in_bad = batch_size
    burn_in_bad = 1000000
    burn_in_expert = 10000000
    batch_size = 400

    gen_expert_data(expert_one_path, expert=True, camera_angle=0, burn_in=burn_in_expert,
                    save_steps=save_steps, batch_size=batch_size)
    #gen_expert_data(expert_two_path, expert=True, camera_angle=1,
    #                burn_in=burn_in_expert, save_steps=save_steps, batch_size=batch_size)
    #gen_expert_data(bad_one_path, expert=False, camera_angle=0, burn_in=burn_in_bad,
    #                save_steps=save_steps, batch_size=batch_size)
    #gen_expert_data(bad_two_path, expert=False, camera_angle=1, burn_in=burn_in_bad,
    #                save_steps=save_steps, batch_size=batch_size)
