import numpy as np
import torch
from torch.distributions import Normal

from models.rssm import get_feat, stack_states
from utils.misc import flatten_rssm_state


class RandomShooting(object):
    def __init__(self,
                 device,
                 action_space,
                 rollout_model,
                 reward_model,
                 horizon,
                 num_control_samples):

        self.device = device
        self.action_space = action_space
        self.rollout_model = rollout_model
        self.reward_model = reward_model
        self.horizon = horizon
        self.num_control_samples = num_control_samples

        # Initialize action distribution
        self.action_dist = Normal(
            torch.zeros((self.horizon, self.action_space.shape[0]), device=device),
            torch.ones((self.horizon, self.action_space.shape[0]), device=device)
        )

    def get_action(self, state):
        with torch.no_grad():
            # sample actions
            action_candidates = self.action_dist.sample([self.num_control_samples])

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
            best_sim_number = torch.argmax(cumulated_reward)
            best_sequence = action_candidates[best_sim_number]

        # select first action
        best_action = best_sequence[0]
        return best_action.detach().cpu().numpy()
