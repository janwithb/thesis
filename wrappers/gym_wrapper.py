import numpy as np
from gym import core, spaces
from dm_env import specs
from utils.viewer import OpenCVImageViewer


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


class GymWrapper(core.Env):
    """
    Gym interface wrapper for dm_control env wrapped by pixels.Wrapper
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-np.inf, np.inf)

    def __init__(self, env):
        self._env = env
        self._viewer = None

    def __getattr(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        observation_space = _spec_to_box(
            self._env.observation_spec().values()
        )
        return observation_space

    @property
    def action_space(self):
        action_space = _spec_to_box([self._env.action_spec()])
        return action_space

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': time_step.discount}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = time_step.observation
        return obs

    def render(self, mode='human', **kwargs):
        img = self._env.physics.render(**kwargs)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self._viewer is None:
                self._viewer = OpenCVImageViewer()
            self._viewer.imshow(img)
            return self._viewer.isopen
        else:
            raise NotImplementedError
