import torch

from models.decoder_model import ObservationDecoder
from models.encoder_model import ObservationEncoder
from models.reward_model import RewardModel
from models.rssm_model import RecurrentStateSpaceModel
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_


class DreamerBase:
    def __init__(self, logger, replay_buffer, device, args):
        super().__init__()

        self.logger = logger
        self.replay_buffer = replay_buffer
        self.device = device
        self.args = args
        self.model_itr = 0
        self.feature_size = args.stochastic_size + args.deterministic_size

        # encoder model
        self.observation_encoder = ObservationEncoder()

        # decoder model
        self.observation_decoder = ObservationDecoder(self.feature_size)

        # reward model
        self.reward_model = RewardModel(self.feature_size, args.reward_hidden_dim)

        # recurrent state space model
        self.rssm = RecurrentStateSpaceModel(args.stochastic_size, args.deterministic_size, args.action_dim)

        # bundle model parameters
        self.model_params = (list(self.observation_encoder.parameters()) +
                             list(self.observation_decoder.parameters()) +
                             list(self.reward_model.parameters()) +
                             list(self.rssm.parameters()))

        # gpu settings
        self.observation_encoder.to(self.device)
        self.observation_decoder.to(self.device)
        self.reward_model.to(self.device)
        self.rssm.to(self.device)

        # model optimizer
        self.model_optimizer = torch.optim.Adam(self.model_params, lr=args.model_lr, eps=args.model_eps)

    def optimize_model(self):
        # compute model loss
        model_loss, flatten_states, flatten_rnn_hiddens = self.model_loss()

        # take gradient step
        self.model_optimizer.zero_grad()
        model_loss.backward()
        clip_grad_norm_(self.model_params, self.args.grad_clip)
        self.model_optimizer.step()
        self.model_itr += 1

        if self.args.full_tb_log and self.model_itr % self.args.model_log_freq == 0:
            self.observation_encoder.log(self.logger, self.model_itr)
            self.observation_decoder.log(self.logger, self.model_itr)
            self.reward_model.log(self.logger, self.model_itr)
            self.rssm.log(self.logger, self.model_itr)
        return flatten_states, flatten_rnn_hiddens

    def model_loss(self):
        observations, actions, rewards, _ = self.replay_buffer.sample(self.args.batch_size, self.args.chunk_length)
        observations = torch.as_tensor(observations, device=self.device).transpose(0, 1)
        actions = torch.as_tensor(actions, device=self.device).transpose(0, 1)
        rewards = torch.as_tensor(rewards, device=self.device).transpose(0, 1)

        # embed observations
        embedded_observations = self.observation_encoder(observations.reshape(-1, 3, 64, 64))
        embedded_observations = embedded_observations.view(self.args.chunk_length, self.args.batch_size, -1)

        # prepare Tensor to maintain states sequence and rnn hidden states sequence
        states = torch.zeros(self.args.chunk_length,
                             self.args.batch_size,
                             self.args.stochastic_size,
                             device=self.device)
        rnn_hiddens = torch.zeros(self.args.chunk_length,
                                  self.args.batch_size,
                                  self.args.deterministic_size,
                                  device=self.device)

        # initialize state and rnn hidden state with 0 vector
        state = torch.zeros(self.args.batch_size, self.args.stochastic_size, device=self.device)
        rnn_hidden = torch.zeros(self.args.batch_size, self.args.deterministic_size, device=self.device)

        # compute state and rnn hidden sequences and kl loss
        kl_loss = 0
        for l in range(self.args.chunk_length - 1):
            next_state_prior, next_state_posterior, rnn_hidden = self.rssm(state,
                                                                           actions[l],
                                                                           rnn_hidden,
                                                                           embedded_observations[l + 1])
            state = next_state_posterior.rsample()
            states[l + 1] = state
            rnn_hiddens[l + 1] = rnn_hidden
            kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
            kl_loss += kl.clamp(min=self.args.free_nats).mean()
        kl_loss /= (self.args.chunk_length - 1)

        # compute reconstructed observations and predicted rewards
        flatten_states = states.view(-1, self.args.stochastic_size)
        flatten_rnn_hiddens = rnn_hiddens.view(-1, self.args.deterministic_size)
        recon_observations = self.observation_decoder(flatten_states, flatten_rnn_hiddens)
        recon_observations = recon_observations.view(self.args.chunk_length, self.args.batch_size, 3, 64, 64)
        predicted_rewards = self.reward_model(flatten_states, flatten_rnn_hiddens)
        predicted_rewards = predicted_rewards.view(self.args.chunk_length, self.args.batch_size, 1)

        # compute loss for observation and reward
        obs_loss = 0.5 * mse_loss(recon_observations[1:], observations[1:], reduction='none').mean([0, 1]).sum()
        reward_loss = 0.5 * mse_loss(predicted_rewards[1:], rewards[:-1])

        # add all losses and update model parameters with gradient descent
        model_loss = self.args.kl_scale * kl_loss + obs_loss + reward_loss

        # log losses
        with torch.no_grad():
            if self.model_itr % self.args.model_log_freq == 0:
                self.logger.log('train_model/observation_loss', obs_loss.item(), self.model_itr)
                self.logger.log('train_model/reward_loss', reward_loss.item(), self.model_itr)
                self.logger.log('train_model/kl_loss', kl_loss.item(), self.model_itr)
                self.logger.log('train_model/overall_loss', model_loss.item(), self.model_itr)
        return model_loss, flatten_states, flatten_rnn_hiddens
