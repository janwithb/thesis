import torch

from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class RecurrentStateSpaceModel(nn.Module):
    """
    This class includes multiple components:
    - Deterministic state model
    - Stochastic state model (prior)
    - State posterior
    """
    def __init__(self, state_dim, rnn_hidden_dim, action_dim,
                 hidden_dim=200, min_stddev=0.1, act=F.relu):
        super(RecurrentStateSpaceModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_rnn_hidden_embedded_obs = nn.Linear(rnn_hidden_dim + 1024, hidden_dim)
        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self._min_stddev = min_stddev
        self.act = act
        self._outputs = dict()

    def forward(self, state, action, rnn_hidden, embedded_next_obs):
        next_state_prior, rnn_hidden = self.prior(state, action, rnn_hidden)
        self._outputs['next_state_prior_mean'] = next_state_prior.mean
        self._outputs['next_state_prior_std'] = next_state_prior.stddev
        self._outputs['rnn_hidden'] = rnn_hidden
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        self._outputs['next_state_posterior_mean'] = next_state_posterior.mean
        self._outputs['next_state_posterior_std'] = next_state_posterior.stddev
        # return prior and posterior for model training
        return next_state_prior, next_state_posterior, rnn_hidden

    def prior(self, state, action, rnn_hidden):
        # compute prior
        hidden = self.act(self.fc_state_action(torch.cat([state, action], dim=1)))
        rnn_hidden = self.rnn(hidden, rnn_hidden)
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))

        mean = self.fc_state_mean_prior(hidden)
        stddev = F.softplus(self.fc_state_stddev_prior(hidden)) + self._min_stddev
        return Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        # compute posterior
        hidden = self.act(self.fc_rnn_hidden_embedded_obs(
            torch.cat([rnn_hidden, embedded_obs], dim=1)))
        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        return Normal(mean, stddev)

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram(f'train_rssm/{k}_hist', v, step)

        logger.log_param('train_rssm/fc_state_action', self.fc_state_action, step)
        logger.log_param('train_rssm/fc_rnn_hidden', self.fc_rnn_hidden, step)
        logger.log_param('train_rssm/fc_state_mean_prior', self.fc_state_mean_prior, step)
        logger.log_param('train_rssm/fc_state_stddev_prior', self.fc_state_stddev_prior, step)
        logger.log_param('train_rssm/fc_rnn_hidden_embedded_obs', self.fc_rnn_hidden_embedded_obs, step)
        logger.log_param('train_rssm/fc_state_mean_posterior', self.fc_state_mean_posterior, step)
        logger.log_param('train_rssm/fc_state_stddev_posterior', self.fc_state_stddev_posterior, step)
