import numpy as np

from PIL import Image
from gym import spaces
from gym import ObservationWrapper


class PixelObservationWrapper(ObservationWrapper):
    def __init__(self, env, preprocessing=True, observation_size=100):
        super(PixelObservationWrapper, self).__init__(env)

        # extend observation space with pixels
        env.reset()
        pixels = self.env.render(mode='rgb_array')
        if np.issubdtype(pixels.dtype, np.integer):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float):
            low, high = (-float('inf'), float('inf'))
        else:
            raise TypeError(pixels.dtype)
        pixels_space = spaces.Box(shape=pixels.shape, low=low, high=high, dtype=pixels.dtype)
        self.observation_space = pixels_space

        self._env = env
        self._preprocessing = preprocessing
        self._observation_size = observation_size

    def observation(self, observation):
        pixel_observation = self._env.render(mode='rgb_array')
        if self._preprocessing:
            prepocessed_observation = self._preprocess_observation(pixel_observation)
            return prepocessed_observation
        else:
            return pixel_observation

    def _preprocess_observation(self, observation):
        prepocessed_observation = self._crop_center_observation(observation)
        prepocessed_observation = self._resize_observation(prepocessed_observation)
        prepocessed_observation = self._grayscale_observation(prepocessed_observation)
        prepocessed_observation = self._normalize_observation(prepocessed_observation)
        return prepocessed_observation

    def _crop_center_observation(self, observation):
        pil_img = Image.fromarray(observation)
        crop_width, crop_height = min(pil_img.size), min(pil_img.size)
        img_width, img_height = pil_img.size
        cropped_pil_img = pil_img.crop(((img_width - crop_width) // 2,
                                        (img_height - crop_height) // 2,
                                        (img_width + crop_width) // 2,
                                        (img_height + crop_height) // 2))
        return np.asarray(cropped_pil_img)

    def _resize_observation(self, observation):
        pil_img = Image.fromarray(observation)
        resized_pil_img = pil_img.resize((self._observation_size, self._observation_size))
        return np.asarray(resized_pil_img)

    def _grayscale_observation(self, observation):
        pil_img = Image.fromarray(observation)
        grayscale_pil_img = pil_img.convert('L')
        return np.asarray(grayscale_pil_img)

    def _normalize_observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0
