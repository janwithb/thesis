import torch
import torch.nn as nn
import numpy as np

from utils.misc import conv_out_shape


class ObservationEncoder(nn.Module):
    def __init__(self,
                 depth=32,
                 stride=2,
                 kernel_size=4,
                 shape=(3, 64, 64),
                 activation=nn.ReLU()):
        super().__init__()

        self.conv1 = nn.Conv2d(shape[0], 1 * depth, kernel_size, stride)
        self.conv2 = nn.Conv2d(1 * depth, 2 * depth, kernel_size, stride)
        self.conv3 = nn.Conv2d(2 * depth, 4 * depth, kernel_size, stride)
        self.conv4 = nn.Conv2d(4 * depth, 8 * depth, kernel_size, stride)

        self._depth = depth
        self._stride = stride
        self._kernel_size = kernel_size
        self._shape = shape
        self._activation = activation
        self._outputs = dict()

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        obs = obs.reshape(-1, *img_shape)
        self._outputs['obs'] = obs
        hidden = self._activation(self.conv1(obs))
        self._outputs['conv1'] = hidden
        hidden = self._activation(self.conv2(hidden))
        self._outputs['conv2'] = hidden
        hidden = self._activation(self.conv3(hidden))
        self._outputs['conv3'] = hidden
        embed = self._activation(self.conv4(hidden))
        embed = torch.reshape(embed, (*batch_shape, -1))
        self._outputs['embed'] = embed
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self._shape[1:], 0, self._kernel_size, self._stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, self._kernel_size, self._stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, self._kernel_size, self._stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, self._kernel_size, self._stride)
        embed_size = 8 * self._depth * np.prod(conv4_shape).item()
        return embed_size

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 3:
                logger.log_image('train_encoder/%s_img' % k, v[0], step)

        logger.log_param('train_encoder/conv1', self.conv1, step)
        logger.log_param('train_encoder/conv2', self.conv2, step)
        logger.log_param('train_encoder/conv3', self.conv3, step)
        logger.log_param('train_encoder/conv4', self.conv4, step)
