import numpy as np

from torch import nn
from torch.nn import functional as F
from utils.misc import conv_out_shape


class ObservationEncoder(nn.Module):
    """
    Encoder to embed image observation (3, 64, 64) to vector (1024,)
    """
    def __init__(self):
        super(ObservationEncoder, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self._outputs = dict()

    def get_embed_size(self):
        conv1_shape = conv_out_shape((64, 64), 0, 4, 2)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, 2)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, 2)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, 2)
        embed_size = 8 * 32 * np.prod(conv4_shape).item()
        return embed_size

    def forward(self, obs):
        self._outputs['obs'] = obs
        hidden = F.relu(self.cv1(obs))
        self._outputs['cv1'] = hidden
        hidden = F.relu(self.cv2(hidden))
        self._outputs['cv2'] = hidden
        hidden = F.relu(self.cv3(hidden))
        self._outputs['cv3'] = hidden
        embedded_obs = F.relu(self.cv4(hidden)).reshape(hidden.size(0), -1)
        self._outputs['cv4'] = embedded_obs
        return embedded_obs

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 3:
                logger.log_image('train_encoder/%s_img' % k, v[0], step)

        logger.log_param('train_encoder/conv1', self.cv1, step)
        logger.log_param('train_encoder/conv2', self.cv2, step)
        logger.log_param('train_encoder/conv3', self.cv3, step)
        logger.log_param('train_encoder/conv4', self.cv4, step)
