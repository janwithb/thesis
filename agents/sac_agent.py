import torch


class PolicyAgent(object):
    def __init__(self, device, rssm, observation_encoder, action_model):
        self.device = device
        self.rssm = rssm
        self.observation_encoder = observation_encoder
        self.action_model = action_model
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

        action = action.squeeze().cpu().numpy()
        return action

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)


def policy(self, state, sample=False):
    feat = get_feat(state)
    dist = self.actor(feat)
    action = dist.sample() if sample else dist.mean
    action = action.clamp(*self.action_range)
    return action, dist


def exploration_policy(self, state):
    if self.training:
        action, _ = self.policy(state, sample=True)
    elif self.eval:
        action, _ = self.policy(state)
    action = np.squeeze(action.detach().cpu().numpy(), axis=0)
    return action