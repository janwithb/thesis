import torch

from torch import nn
from torch.nn import functional as F


class ValueModel(nn.Module):
    """
    Value model to predict state-value of current policy (action_model).
    """
    def __init__(self, feature_size, hidden_dim=400, act=F.elu):
        super(ValueModel, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act
        self._outputs = dict()

    def forward(self, state, rnn_hidden):
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        self._outputs['fc1'] = hidden
        hidden = self.act(self.fc2(hidden))
        self._outputs['fc2'] = hidden
        hidden = self.act(self.fc3(hidden))
        self._outputs['fc3'] = hidden
        state_value = self.fc4(hidden)
        self._outputs['state_value'] = state_value
        return state_value

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram(f'train_value_model/{k}_hist', v, step)

        logger.log_param('train_value_model/fc1', self.fc1, step)
        logger.log_param('train_value_model/fc2', self.fc2, step)
        logger.log_param('train_value_model/fc3', self.fc3, step)
        logger.log_param('train_value_model/fc4', self.fc4, step)
