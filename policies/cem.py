import numpy as np
import torch

from models.rssm import get_feat, stack_states
from utils.misc import flatten_rssm_state


class CEM(object):

    def __init__(self,
                 action_space,
                 rollout_model,
                 reward_model,
                 horizon,
                 num_control_samples,
                 max_iterations,
                 num_elites):

        self.action_space = action_space
        self.rollout_model = rollout_model
        self.reward_model = reward_model
        self.horizon = horizon
        self.num_control_samples = num_control_samples
        self.max_iterations = max_iterations
        self.num_elites = num_elites
        self.mu = np.zeros(self.horizon * self.action_space.shape[0])
        self.sigma = np.array([0.5 ** 0.5] * (self.horizon * self.action_space.shape[0]))

    def get_action(self, state, device):
        mu = self.mu
        sigma = self.sigma

        for i in range(self.max_iterations):
            # sample actions
            original_shape = (self.num_control_samples, self.horizon, self.action_space.shape[0])
            sample_shape = (self.num_control_samples, self.horizon * self.action_space.shape[0])
            samples = np.float32(np.random.normal(mu, sigma, size=sample_shape))
            samples = np.clip(samples, a_min=self.action_space.low[0], a_max=self.action_space.high[0])

            # prepare samples
            actions = torch.as_tensor(samples, device=device)
            actions = torch.reshape(actions, original_shape)

            # prepare initial states
            initial_states = np.full(self.num_control_samples, state).tolist()
            initial_states = stack_states(initial_states, dim=0)
            initial_states = flatten_rssm_state(initial_states)

            # predict resulting states
            prior = self.rollout_model.rollout_transition(self.horizon, actions, initial_states)
            feat = get_feat(prior)

            # predict rewards
            reward_pred = self.reward_model(feat).mean
            cumulated_reward = torch.sum(reward_pred, dim=1)

            # pick best action sequence
            sorted_sim_numbers = torch.squeeze(torch.argsort(cumulated_reward, dim=0, descending=True))
            elite_sim_numbers = sorted_sim_numbers[:self.num_elites]
            elite_actions = samples[elite_sim_numbers, :]
            mu = np.mean(elite_actions, 0)
            sigma = np.std(elite_actions, 0)
        best_sample = np.reshape(mu, (self.horizon, self.action_space.shape[0]))[0]

        return np.array(best_sample)
