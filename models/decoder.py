import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np

from utils.misc import conv_out_shape, output_padding_shape


class ObservationDecoder(nn.Module):
    def __init__(self,
                 depth=32,
                 stride=2,
                 activation=nn.ReLU,
                 embed_size=1024,
                 shape=(3, 64, 64)):
        super().__init__()

        self.depth = depth
        self.shape = shape

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
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, conv4_kernel_size, stride, output_padding=conv4_pad),
            activation(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, conv3_kernel_size, stride, output_padding=conv3_pad),
            activation(),
            nn.ConvTranspose2d(2 * depth, 1 * depth, conv2_kernel_size, stride, output_padding=conv2_pad),
            activation(),
            nn.ConvTranspose2d(1 * depth, shape[0], conv1_kernel_size, stride, output_padding=conv1_pad),
        )

    def forward(self, x):
        """
        :param x: size(*batch_shape, embed_size)
        :return: obs_dist = size(*batch_shape, *self.shape)
        """
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.shape))
        return obs_dist
