import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from utils.distribution import TanhBijector, SampleDist


class ActorModel(nn.Module):
    def __init__(self,
                 action_size,
                 feature_size,
                 hidden_size,
                 layers,
                 dist='tanh_normal',
                 activation=nn.ELU,
                 min_std=1e-4,
                 init_std=5,
                 mean_scale=5):
        super().__init__()

        self._action_size = action_size
        self._feature_size = feature_size
        self._hidden_size = hidden_size
        self._layers = layers
        self._dist = dist
        self._activation = activation
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale
        self._feedforward_model = self.build_model()
        self._raw_init_std = np.log(np.exp(self._init_std) - 1)
        self._outputs = dict()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self._activation()]
        for i in range(1, self._layers):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self._activation()]
        if self._dist == 'tanh_normal':
            model += [nn.Linear(self._hidden_size, self._action_size * 2)]
        elif self._dist == 'one_hot' or self._dist == 'relaxed_one_hot':
            model += [nn.Linear(self._hidden_size, self._action_size)]
        else:
            raise NotImplementedError(f'{self._dist} not implemented')
        return nn.Sequential(*model)

    def forward(self, state_features):
        x = self._feedforward_model(state_features)
        self._outputs['dist_input'] = x
        dist = None
        if self._dist == 'tanh_normal':
            mean, std = torch.chunk(x, 2, -1)
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            std = F.softplus(std + self._raw_init_std) + self._min_std
            dist = torch.distributions.Normal(mean, std)
            dist = torch.distributions.TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == 'one_hot':
            dist = torch.distributions.OneHotCategorical(logits=x)
        elif self._dist == 'relaxed_one_hot':
            dist = torch.distributions.RelaxedOneHotCategorical(0.1, logits=x)
        return dist

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self._feedforward_model):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)
