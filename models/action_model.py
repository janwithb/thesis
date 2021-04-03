import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class ActionModel(nn.Module):
    """
    Action model to compute action from state and rnn_hidden
    """
    def __init__(self, feature_size, action_dim, hidden_dim=400, act=F.elu, min_stddev=1e-4, init_stddev=5.0):
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_stddev = nn.Linear(hidden_dim, action_dim)
        self.act = act
        self.min_stddev = min_stddev
        self.init_stddev = np.log(np.exp(init_stddev) - 1)
        self._outputs = dict()

    def forward(self, state, rnn_hidden, exploration=True):
        """
        if training=True, returned action is reparametrized sample
        if training=False, returned action is mean of action distribution
        """
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        self._outputs['fc1'] = hidden
        hidden = self.act(self.fc2(hidden))
        self._outputs['fc2'] = hidden
        hidden = self.act(self.fc3(hidden))
        self._outputs['fc3'] = hidden
        hidden = self.act(self.fc4(hidden))
        self._outputs['fc4'] = hidden

        # action-mean is divided by 5.0 and applied tanh and scaled by 5.0
        mean = self.fc_mean(hidden)
        mean = 5.0 * torch.tanh(mean / 5.0)
        self._outputs['mean'] = mean

        # compute stddev
        stddev = self.fc_stddev(hidden)
        stddev = F.softplus(stddev + self.init_stddev) + self.min_stddev
        self._outputs['stddev'] = stddev

        if exploration:
            action = torch.tanh(Normal(mean, stddev).rsample())
        else:
            action = torch.tanh(mean)
        return action

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram(f'train_action_model/{k}_hist', v, step)

        logger.log_param('train_action_model/fc1', self.fc1, step)
        logger.log_param('train_action_model/fc2', self.fc2, step)
        logger.log_param('train_action_model/fc3', self.fc3, step)
        logger.log_param('train_action_model/fc4', self.fc4, step)
