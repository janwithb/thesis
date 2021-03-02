import numpy as np

from PIL import Image
from gym import spaces
from gym import ObservationWrapper


class PixelObservationWrapper(ObservationWrapper):
    def __init__(self,
                 env,
                 crop_center_observation=True,
                 resize_observation=True,
                 observation_size=100,
                 grayscale_observation=True,
                 normalize_observation=True):
        super(PixelObservationWrapper, self).__init__(env)

        self._env = env
        self._crop_center_observation = crop_center_observation
        self._resize_observation = resize_observation
        self._observation_size = observation_size
        self._grayscale_observation = grayscale_observation
        self._normalize_observation = normalize_observation

        # extend observation space with pixels
        env.reset()
        pixels = self.preprocess_observation(self.env.render(mode='rgb_array'))
        if np.issubdtype(pixels.dtype, np.integer):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float32):
            low, high = (np.float32(0), np.float32(1))
        else:
            raise TypeError(pixels.dtype)
        pixels_space = spaces.Box(shape=pixels.shape, low=low, high=high, dtype=pixels.dtype)
        self.observation_space = pixels_space

    def observation(self, observation):
        pixel_observation = self._env.render(mode='rgb_array')
        prepocessed_observation = self.preprocess_observation(pixel_observation)
        return prepocessed_observation

    def preprocess_observation(self, observation):
        prepocessed_observation = observation
        if self._crop_center_observation:
            prepocessed_observation = self.crop_center_observation(observation)
        if self._resize_observation:
            prepocessed_observation = self.resize_observation(prepocessed_observation)
        if self._grayscale_observation:
            prepocessed_observation = self.grayscale_observation(prepocessed_observation)
        if self._normalize_observation:
            prepocessed_observation = self.normalize_observation(prepocessed_observation)

        # change shape from (height, width, channels) to (channels, height, width)
        prepocessed_observation = np.rollaxis(prepocessed_observation, 2, 0)
        return prepocessed_observation

    def crop_center_observation(self, observation):
        pil_img = Image.fromarray(observation)
        crop_width, crop_height = min(pil_img.size), min(pil_img.size)
        img_width, img_height = pil_img.size
        cropped_pil_img = pil_img.crop(((img_width - crop_width) // 2,
                                        (img_height - crop_height) // 2,
                                        (img_width + crop_width) // 2,
                                        (img_height + crop_height) // 2))
        return np.asarray(cropped_pil_img)

    def resize_observation(self, observation):
        pil_img = Image.fromarray(observation)
        resized_pil_img = pil_img.resize((self._observation_size, self._observation_size))
        return np.asarray(resized_pil_img)

    def grayscale_observation(self, observation):
        pil_img = Image.fromarray(observation)
        grayscale_pil_img = np.expand_dims(np.asarray(pil_img.convert('L')), 2)
        return grayscale_pil_img

    def normalize_observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0
