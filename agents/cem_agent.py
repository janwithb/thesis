import torch
import numpy as np

from torch.distributions import Normal


class CEM(object):
    def __init__(self,
                 device,
                 action_dim,
                 observation_encoder,
                 reward_model,
                 rssm,
                 horizon,
                 num_control_samples,
                 max_iterations,
                 num_elites,
                 exploration_noise_var):

        self.device = device
        self.action_dim = action_dim
        self.observation_encoder = observation_encoder
        self.reward_model = reward_model
        self.rssm = rssm
        self.horizon = horizon
        self.num_control_samples = num_control_samples
        self.max_iterations = max_iterations
        self.num_elites = num_elites
        self.exploration_noise_var = exploration_noise_var
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=device)

    def get_action(self, obs, exploration=False):
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.unsqueeze(0)

        with torch.no_grad():
            # encode observation
            embedded_obs = self.observation_encoder(obs)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)

            # initialize action distribution
            action_dist = Normal(
                torch.zeros((self.horizon, self.action_dim), device=self.device),
                torch.ones((self.horizon, self.action_dim), device=self.device)
            )

            for itr in range(self.max_iterations):
                # sample actions
                action_candidates = action_dist.sample([self.num_control_samples]).transpose(0, 1)

                # initialize reward, state, and rnn hidden state
                total_predicted_reward = torch.zeros(self.num_control_samples, device=self.device)
                state = state_posterior.sample([self.num_control_samples]).squeeze()
                rnn_hidden = self.rnn_hidden.repeat([self.num_control_samples, 1])

                # compute total predicted reward
                for t in range(self.horizon):
                    next_state_prior, rnn_hidden = self.rssm.prior(state, action_candidates[t], rnn_hidden)
                    state = next_state_prior.sample()
                    total_predicted_reward += self.reward_model(state, rnn_hidden).squeeze()

                # update action distribution using top-k samples
                top_indexes = total_predicted_reward.argsort(descending=True)[: self.num_elites]
                top_action_candidates = action_candidates[:, top_indexes, :]
                mean = top_action_candidates.mean(dim=1)
                stddev = (top_action_candidates - mean.unsqueeze(1)).abs().sum(dim=1) / (self.num_elites - 1)
                action_dist = Normal(mean, stddev)

            # return only first action
            action = mean[0]
            expl_std = torch.sqrt(torch.as_tensor(self.exploration_noise_var, device=self.device))
            expl_mean = torch.tensor(0).to(device=self.device)

            # exploration
            if exploration:
                action += torch.normal(expl_mean, expl_std, size=(self.action_dim,))

            # update rnn hidden state for next step planning
            _, self.rnn_hidden = self.rssm.prior(state_posterior.sample(),
                                                 action.unsqueeze(0),
                                                 self.rnn_hidden)

        action = action.cpu().numpy()
        return action

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
