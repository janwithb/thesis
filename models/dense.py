import torch
import torch.distributions as td
import torch.nn as nn
import numpy as np


class DenseModel(nn.Module):
    def __init__(self,
                 feature_size,
                 output_shape,
                 layers,
                 hidden_size,
                 model_log_name='dense',
                 dist='normal',
                 activation=nn.ELU):
        super().__init__()

        self._feature_size = feature_size
        self._output_shape = output_shape
        self._layers = layers
        self._hidden_size = hidden_size
        self._model_log_name = model_log_name
        self._dist = dist
        self._activation = activation
        self._outputs = dict()

        # build model structure
        self._model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self._activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self._activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self._model(features)
        self._outputs['dist_input'] = dist_inputs
        reshaped_inputs = torch.reshape(dist_inputs, features.shape[:-1] + self._output_shape)
        if self._dist == 'normal':
            return td.independent.Independent(td.Normal(reshaped_inputs, 1), len(self._output_shape))
        elif self._dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=reshaped_inputs), len(self._output_shape))
        elif self._dist == 'scalar':
            return dist_inputs
        else:
            raise NotImplementedError(self._dist)

    def log(self, logger, step):
        for k, v in self._outputs.items():
            logger.log_histogram('train_' + self._model_log_name + f'/{k}_hist', v, step)

        for i, m in enumerate(self._model):
            if type(m) == nn.Linear:
                logger.log_param('train_' + self._model_log_name + f'/fc{i}', m, step)
