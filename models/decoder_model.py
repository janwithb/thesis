import torch

from torch import nn
from torch.nn import functional as F


class ObservationDecoder(nn.Module):
    """
    Observation model to reconstruct image observation (3, 64, 64).
    """
    def __init__(self, feature_size):
        super(ObservationDecoder, self).__init__()
        self.fc = nn.Linear(feature_size, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)
        self._outputs = dict()

    def forward(self, state, rnn_hidden):
        feature = torch.cat([state, rnn_hidden], dim=1)
        self._outputs['feature'] = feature
        hidden = self.fc(feature)
        self._outputs['fc'] = hidden
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        self._outputs['dc1'] = hidden
        hidden = F.relu(self.dc2(hidden))
        self._outputs['dc2'] = hidden
        hidden = F.relu(self.dc3(hidden))
        self._outputs['dc3'] = hidden
        obs = self.dc4(hidden)
        self._outputs['dc4'] = obs
        return obs

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 3 and k != 'linear':
                logger.log_image('train_decoder/%s_img' % k, v[0], step)

        logger.log_param('train_decoder/linear', self.fc, step)
        logger.log_param('train_decoder/dc1', self.dc1, step)
        logger.log_param('train_decoder/dc2', self.dc2, step)
        logger.log_param('train_decoder/dc3', self.dc3, step)
        logger.log_param('train_decoder/dc4', self.dc4, step)
