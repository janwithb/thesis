import numpy as np
import torch

from models.rssm import get_feat, stack_states
from utils.misc import flatten_rssm_state


class RandomShooting(object):

    def __init__(self,
                 action_space,
                 rollout_model,
                 reward_model,
                 horizon,
                 num_control_samples):

        self.action_space = action_space
        self.rollout_model = rollout_model
        self.reward_model = reward_model
        self.horizon = horizon
        self.num_control_samples = num_control_samples
        self.mu = np.zeros(self.horizon * self.action_space.shape[0])
        self.sigma = np.array([0.5 ** 0.5] * (self.horizon * self.action_space.shape[0]))

    def get_action(self, state, device):
        shape = (self.num_control_samples, self.horizon * self.action_space.shape[0])
        actions = np.float32(np.random.normal(self.mu, self.sigma, size=shape))
        actions = np.clip(actions, a_min=self.action_space.low, a_max=self.action_space.high)
        actions = torch.as_tensor(actions, device=device)
        actions = torch.unsqueeze(actions, dim=2)

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
        best_sim_number = torch.argmax(cumulated_reward)
        best_sequence = actions[best_sim_number]
        best_action = best_sequence[0].detach().cpu().numpy()

        return best_action
