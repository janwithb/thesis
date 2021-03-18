import numpy as np
import torch

from torch.distributions import Normal
from models.rssm import get_feat, stack_states
from utils.misc import flatten_rssm_state


class CEM(object):

    def __init__(self,
                 device,
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

        # Initialize action distribution
        self.action_dist = Normal(
            torch.zeros((self.horizon, self.action_space.shape[0]), device=device),
            torch.ones((self.horizon, self.action_space.shape[0]), device=device)
        )

    def get_action(self, state):
        with torch.no_grad():
            action_dist = self.action_dist

            for i in range(self.max_iterations):
                # sample actions
                action_candidates = action_dist.sample([self.num_control_samples])

                # prepare initial states
                initial_states = np.full(self.num_control_samples, state).tolist()
                initial_states = stack_states(initial_states, dim=0)
                initial_states = flatten_rssm_state(initial_states)

                # predict resulting states
                prior = self.rollout_model.rollout_transition(self.horizon, action_candidates, initial_states)
                feat = get_feat(prior)

                # predict rewards
                reward_pred = self.reward_model(feat).mean
                cumulated_reward = torch.sum(reward_pred, dim=1)

                # pick best action sequence
                sorted_sim_numbers = torch.squeeze(torch.argsort(cumulated_reward, dim=0, descending=True))
                elite_sim_numbers = sorted_sim_numbers[:self.num_elites]
                elite_actions = action_candidates[elite_sim_numbers, :]

                # update distribution
                mean = elite_actions.mean(dim=0)
                stddev = (elite_actions - mean.unsqueeze(0)).abs().sum(dim=0) / (self.num_elites - 1)
                action_dist = Normal(mean, stddev)

        # select first action
        best_sample = mean[0]
        return best_sample.detach().cpu().numpy()
