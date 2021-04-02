import numpy as np

from gym import spaces
from gym import ObservationWrapper


class PixelObservation(ObservationWrapper):
    def __init__(self, env, observation_size):
        super(PixelObservation, self).__init__(env)
        self.env = env
        self._observation_size = observation_size

        # extend observation space with pixels
        pixels = self.preprocess_observation(self.env.render(mode='rgb_array',
                                                             height=observation_size,
                                                             width=observation_size,
                                                             camera_id=0))

        # set observation space
        if np.issubdtype(pixels.dtype, np.integer):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float32):
            low, high = (-np.float32(1), np.float32(1))
        else:
            raise TypeError(pixels.dtype)
        pixels_space = spaces.Box(shape=pixels.shape, low=low, high=high, dtype=pixels.dtype)
        self.observation_space = pixels_space

    def observation(self, observation):
        pixel_observation = self.env.render(mode='rgb_array',
                                            height=self._observation_size,
                                            width=self._observation_size,
                                            camera_id=0)
        prepocessed_observation = self.preprocess_observation(pixel_observation)
        return prepocessed_observation

    def preprocess_observation(self, observation):
        # normalize observation
        prepocessed_observation = np.array(observation).astype(np.float32) / 255.0 - 0.5

        # change shape from (height, width, channels) to (channels, height, width)
        prepocessed_observation = np.rollaxis(prepocessed_observation, 2, 0)
        return prepocessed_observation
