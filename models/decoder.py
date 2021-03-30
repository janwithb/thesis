import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np

from utils.misc import conv_out_shape, output_padding_shape


class ObservationDecoder(nn.Module):
    def __init__(self,
                 depth=32,
                 stride=2,
                 embed_size=1024,
                 shape=(3, 64, 64),
                 activation=nn.ReLU()):
        super().__init__()

        c, h, w = shape
        conv1_kernel_size = 6
        conv2_kernel_size = 6
        conv3_kernel_size = 5
        conv4_kernel_size = 5
        padding = 0
        conv1_shape = conv_out_shape((h, w), padding, conv1_kernel_size, stride)
        conv1_pad = output_padding_shape((h, w), conv1_shape, padding, conv1_kernel_size, stride)
        conv2_shape = conv_out_shape(conv1_shape, padding, conv2_kernel_size, stride)
        conv2_pad = output_padding_shape(conv1_shape, conv2_shape, padding, conv2_kernel_size, stride)
        conv3_shape = conv_out_shape(conv2_shape, padding, conv3_kernel_size, stride)
        conv3_pad = output_padding_shape(conv2_shape, conv3_shape, padding, conv3_kernel_size, stride)
        conv4_shape = conv_out_shape(conv3_shape, padding, conv4_kernel_size, stride)
        conv4_pad = output_padding_shape(conv3_shape, conv4_shape, padding, conv4_kernel_size, stride)

        self.conv_shape = (32 * depth, *conv4_shape)
        self.linear = nn.Linear(embed_size, 32 * depth * np.prod(conv4_shape).item())
        self.dc1 = nn.ConvTranspose2d(32 * depth, 4 * depth, conv4_kernel_size, stride, output_padding=conv4_pad)
        self.dc2 = nn.ConvTranspose2d(4 * depth, 2 * depth, conv3_kernel_size, stride, output_padding=conv3_pad)
        self.dc3 = nn.ConvTranspose2d(2 * depth, 1 * depth, conv2_kernel_size, stride, output_padding=conv2_pad)
        self.dc4 = nn.ConvTranspose2d(1 * depth, shape[0], conv1_kernel_size, stride, output_padding=conv1_pad)

        self._shape = shape
        self._activation = activation
        self._outputs = dict()

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        self._outputs['embed'] = x
        hidden = self.linear(x)
        hidden = torch.reshape(hidden, (squeezed_size, *self.conv_shape))
        self._outputs['linear'] = hidden
        hidden = self._activation(self.dc1(hidden))
        self._outputs['dc1'] = hidden
        hidden = self._activation(self.dc2(hidden))
        self._outputs['dc2'] = hidden
        hidden = self._activation(self.dc3(hidden))
        self._outputs['dc3'] = hidden
        hidden = self.dc4(hidden)
        self._outputs['dc4'] = hidden
        mean = torch.reshape(hidden, (*batch_shape, *self._shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self._shape))
        return obs_dist

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 3 and k != 'linear':
                logger.log_image('train_decoder/%s_img' % k, v[0], step)

        logger.log_param('train_decoder/linear', self.linear, step)
        logger.log_param('train_decoder/dc1', self.dc1, step)
        logger.log_param('train_decoder/dc2', self.dc2, step)
        logger.log_param('train_decoder/dc3', self.dc3, step)
        logger.log_param('train_decoder/dc4', self.dc4, step)
