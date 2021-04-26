import torch
import numpy as np


class SACAgent(object):
    def __init__(self, device, action_range, rssm, observation_encoder, actor):
        self.device = device
        self.action_range = action_range
        self.rssm = rssm
        self.observation_encoder = observation_encoder
        self.actor = actor
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=device)

    def get_action(self, obs, exploration=False):
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.unsqueeze(0)

        with torch.no_grad():
            # encode observation
            embedded_obs = self.observation_encoder(obs)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample()
            feature = torch.cat([state, self.rnn_hidden], dim=1)
            dist = self.actor(feature)

            # exploration
            if exploration:
                action = dist.sample()
            else:
                action = dist.mean
            action = action.clamp(*self.action_range)

            # update rnn_hidden for next step
            _, self.rnn_hidden = self.rssm.prior(state, action, self.rnn_hidden)

        action = action.squeeze().cpu().numpy()
        if np.ndim(action) == 0:
            action = np.array([action.tolist()])
        return action

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
