from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
import numpy as np
import math
from rllab.mujoco_py import glfw


class PointEnv(MujocoEnv, Serializable):

    """
    Use Left, Right, Up, Down, A (steer left), D (steer right)
    """

    FILE = 'blue_point.xml'

    def __init__(self, *args, **kwargs):
        super(PointEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.image_height = 200
        self.image_width = 200
        self.save_steps = 1000
        self.burn_in_steps = 2400
        self.image_history = np.zeros(shape=(self.save_steps, self.image_height, self.image_width, 3))
        self.image_history_step = 0
        self.goal = [0, 0]

        save_dir = '/Users/TheMaster/Desktop/Current_Work/irl/irl/experiments/pointmass/domain_one_failure'
        reward_function = '0'
        fixed_camera = None

        self.save_dir = save_dir
        if reward_function == 'standard':
            self.reward_function = self.standard_reward
        elif reward_function == '0':
            self.reward_function = self.zero_reward
        elif reward_function == 'gan':
            self.reward_function = self.gan_reward
        self.fixed_camera = fixed_camera

        self.gan_discriminator = None

    def zero_reward(self):
        return 0

    def standard_reward(self):
        pass

    def gan_reward(self):
        self.gan_discriminator(self.image_history[-1])

    def step(self, action):
        qpos = np.copy(self.model.data.qpos)
        qpos[2, 0] += action[1]
        ori = qpos[2, 0]
        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        clip_point = 4
        qpos[0, 0] = np.clip(qpos[0, 0] + dx, -clip_point, clip_point)
        qpos[1, 0] = np.clip(qpos[1, 0] + dy, -clip_point, clip_point)
        reward = np.abs(qpos[0, 0]) + np.abs(qpos[1, 0])
        print reward
        #reward = 0
        cost = -reward
        self.model.data.qpos = qpos
        self.model.forward()
        next_obs = self.get_current_obs()
        #self.render()

        #self.image_history_step += 1

        #if (self.image_history_step - self.burn_in_steps) == self.save_steps:
        #        np.save('/Users/TheMaster/Desktop/Current_Work/irl/experiments/pointmass/domain_one_failure', self.image_history)
        #        print 'saved all data'
        #        self.stop = True

        #elif self.image_history_step >= self.burn_in_steps and self.stop is False:
        #    if self.image_history_step % 500 == 0:
        #        print self.image_history_step
        #    data, width, height = self.get_viewer().get_image()
        #    tim = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        #    #print tim
        #    self.image_history[self.image_history_step-self.burn_in_steps] = tim
        #    #print self.image_history_step
        return Step(next_obs, cost, False)

    def get_xy(self):
        qpos = self.model.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def set_xy(self, xy):
        qpos = np.copy(self.model.data.qpos)
        qpos[0, 0] = xy[0]
        qpos[1, 0] = xy[1]
        self.model.data.qpos = qpos
        self.model.forward()

    @overrides
    def action_from_key(self, key):
        lb, ub = self.action_bounds
        if key == glfw.KEY_LEFT:
            return np.array([0, ub[0]*0.3])
        elif key == glfw.KEY_RIGHT:
            return np.array([0, lb[0]*0.3])
        elif key == glfw.KEY_UP:
            return np.array([ub[1], 0])
        elif key == glfw.KEY_DOWN:
            return np.array([lb[1], 0])
        else:
            return np.array([0, 0])



