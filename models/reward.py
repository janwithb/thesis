import torch

from torch import nn
from torch.nn import functional as F


class RewardModel(nn.Module):
    """
    p(r_t | s_t, h_t)
    Reward model to predict reward from state and rnn hidden state
    """
    def __init__(self, feature_size, hidden_dim=300, act=F.relu):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act
        self._outputs = dict()

    def forward(self, state, rnn_hidden):
        feature = torch.cat([state, rnn_hidden], dim=1)
        self._outputs['feature'] = feature
        hidden = self.act(self.fc1(feature))
        self._outputs['fc1'] = hidden
        hidden = self.act(self.fc2(hidden))
        self._outputs['fc2'] = hidden
        hidden = self.act(self.fc3(hidden))
        self._outputs['fc3'] = hidden
        reward = self.fc4(hidden)
        self._outputs['reward'] = reward
        return reward

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram(f'train_reward_model/{k}_hist', v, step)

        logger.log_param('train_reward_model/fc1', self.fc1, step)
        logger.log_param('train_reward_model/fc2', self.fc2, step)
        logger.log_param('train_reward_model/fc3', self.fc3, step)
        logger.log_param('train_reward_model/fc4', self.fc4, step)
