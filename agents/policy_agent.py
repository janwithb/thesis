import torch
import numpy as np


class PolicyAgent(object):
    def __init__(self, device, action_dim, rssm, observation_encoder, action_model, exploration_noise_var):
        self.device = device
        self.action_dim = action_dim
        self.rssm = rssm
        self.observation_encoder = observation_encoder
        self.action_model = action_model
        self.exploration_noise_var = exploration_noise_var
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=device)

    def get_action(self, obs, exploration=False):
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.unsqueeze(0)

        with torch.no_grad():
            # encode observation
            embedded_obs = self.observation_encoder(obs)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample()
            action = self.action_model(state, self.rnn_hidden, exploration=exploration)

            # update rnn_hidden for next step
            _, self.rnn_hidden = self.rssm.prior(state, action, self.rnn_hidden)

        # exploration
        action = action.squeeze().cpu().numpy()
        if exploration:
            action += np.random.normal(0, np.sqrt(self.exploration_noise_var), self.action_dim)
        return action

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
