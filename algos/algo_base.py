import torch
import numpy as np

from torch import nn

from models.curl_model import CURLModel
from models.decoder_model import ObservationDecoder
from models.encoder_model import ObservationEncoder
from models.reward_model import RewardModel
from models.rssm_model import RecurrentStateSpaceModel
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from utils.misc import augument_image


class AlgoBase:
    """
    Base algorithm that includes training the latent dynamics model.
    """
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

        # CURL model
        self.embed_size = self.observation_encoder.get_embed_size()
        if self.args.image_loss_type in ('reconstruction', 'obs_embed_contrast', 'aug_obs_embed_contrast'):
            self.curl_model = CURLModel(device, self.embed_size, self.feature_size, similarity=args.similarity,
                                        temperature=args.curl_temperature)
        elif self.args.image_loss_type == 'augment_contrast':
            self.curl_model = CURLModel(device, self.feature_size, self.feature_size, similarity=args.similarity,
                                        temperature=args.curl_temperature)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # bundle model parameters
        self.model_params = (list(self.observation_encoder.parameters()) +
                             list(self.reward_model.parameters()) +
                             list(self.rssm.parameters()))
        if self.args.image_loss_type == 'reconstruction':
            self.model_params = self.model_params + list(self.observation_decoder.parameters())
        elif self.args.image_loss_type in ('obs_embed_contrast', 'augment_contrast', 'aug_obs_embed_contrast'):
            self.model_params = self.model_params + list(self.curl_model.parameters())
        else:
            raise ValueError('unknown image loss type')

        # gpu settings
        self.observation_encoder.to(self.device)
        self.observation_decoder.to(self.device)
        self.reward_model.to(self.device)
        self.rssm.to(self.device)
        self.curl_model.to(self.device)

        # model optimizer
        self.model_optimizer = torch.optim.Adam(self.model_params, lr=args.model_lr, eps=args.model_eps)

    def optimize_model(self):
        # compute model loss
        model_loss, flatten_states, flatten_rnn_hiddens, rewards, actions = self.model_loss()

        # take gradient step
        self.model_optimizer.zero_grad()
        model_loss.backward()
        clip_grad_norm_(self.model_params, self.args.grad_clip)
        self.model_optimizer.step()
        self.model_itr += 1

        if self.args.full_tb_log and (self.model_itr % self.args.model_log_freq == 0):
            self.observation_encoder.log(self.logger, self.model_itr)
            if self.args.image_loss_type == 'reconstruction':
                self.observation_decoder.log(self.logger, self.model_itr)
            self.reward_model.log(self.logger, self.model_itr)
            self.rssm.log(self.logger, self.model_itr)
        return flatten_states, flatten_rnn_hiddens, rewards, actions

    def model_loss(self):
        if self.args.image_loss_type in ('reconstruction', 'obs_embed_contrast'):
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

            # compute observation loss
            if self.args.image_loss_type == 'reconstruction':
                recon_observations = self.observation_decoder(flatten_states, flatten_rnn_hiddens)
                recon_observations = recon_observations.view(self.args.chunk_length, self.args.batch_size, 3, 64, 64)
                obs_loss = 0.5 * mse_loss(recon_observations[1:], observations[1:], reduction='none').mean([0, 1]).sum()
            elif self.args.image_loss_type == 'obs_embed_contrast':
                feature = torch.cat([flatten_states, flatten_rnn_hiddens], dim=1)
                flatten_embeds = embedded_observations.view(-1, self.embed_size)
                logits, labels = self.curl_model.info_nce_loss(feature, flatten_embeds)
                obs_loss = self.cross_entropy_loss(logits, labels)
            predicted_rewards = self.reward_model(flatten_states, flatten_rnn_hiddens)
            predicted_rewards = predicted_rewards.view(self.args.chunk_length, self.args.batch_size, 1)

            # compute loss for observation and reward
            reward_loss = 0.5 * mse_loss(predicted_rewards[1:], rewards[:-1])
        elif self.args.image_loss_type in ('augment_contrast', 'aug_obs_embed_contrast'):
            observations_a, actions, rewards, _ = self.replay_buffer.sample(self.args.batch_size, self.args.chunk_length)
            observations_pos = observations_a.copy()
            observations_a = torch.as_tensor(observations_a, device=self.device).transpose(0, 1)
            observations_pos = torch.as_tensor(observations_pos, device=self.device).transpose(0, 1)
            actions = torch.as_tensor(actions, device=self.device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=self.device).transpose(0, 1)

            # augment images
            size = self.args.observation_size
            observations_a = augument_image(observations_a.reshape(-1, 3, size, size))
            observations_pos = augument_image(observations_pos.reshape(-1, 3, size, size))

            # embed observations
            embedded_observations_a = self.observation_encoder(observations_a)
            embedded_observations_a = embedded_observations_a.view(self.args.chunk_length, self.args.batch_size, -1)
            embedded_observations_pos = self.observation_encoder(observations_pos)
            embedded_observations_pos = embedded_observations_pos.view(self.args.chunk_length, self.args.batch_size, -1)

            # prepare Tensor to maintain states sequence and rnn hidden states sequence
            states_a = torch.zeros(self.args.chunk_length,
                                   self.args.batch_size,
                                   self.args.stochastic_size,
                                   device=self.device)
            states_pos = torch.zeros(self.args.chunk_length,
                                     self.args.batch_size,
                                     self.args.stochastic_size,
                                     device=self.device)
            rnn_hiddens_a = torch.zeros(self.args.chunk_length,
                                        self.args.batch_size,
                                        self.args.deterministic_size,
                                        device=self.device)
            rnn_hiddens_pos = torch.zeros(self.args.chunk_length,
                                          self.args.batch_size,
                                          self.args.deterministic_size,
                                          device=self.device)

            # initialize state and rnn hidden state with 0 vector
            state_a = torch.zeros(self.args.batch_size, self.args.stochastic_size, device=self.device)
            state_pos = torch.zeros(self.args.batch_size, self.args.stochastic_size, device=self.device)
            rnn_hidden_a = torch.zeros(self.args.batch_size, self.args.deterministic_size, device=self.device)
            rnn_hidden_pos = torch.zeros(self.args.batch_size, self.args.deterministic_size, device=self.device)

            # compute state and rnn hidden sequences and kl loss
            kl_loss = 0
            for l in range(self.args.chunk_length - 1):
                next_state_prior_a, next_state_posterior_a, rnn_hidden_a = self.rssm(state_a,
                                                                                     actions[l],
                                                                                     rnn_hidden_a,
                                                                                     embedded_observations_a[l + 1])

                with torch.no_grad():
                    (next_state_prior_pos,
                     next_state_posterior_pos,
                     rnn_hidden_pos) = self.rssm(state_pos,
                                                 actions[l],
                                                 rnn_hidden_pos,
                                                 embedded_observations_pos[l + 1])
                state_a = next_state_posterior_a.rsample()
                state_pos = next_state_posterior_pos.rsample()
                states_a[l + 1] = state_a
                states_pos[l + 1] = state_pos
                rnn_hiddens_a[l + 1] = rnn_hidden_a
                rnn_hiddens_pos[l + 1] = rnn_hidden_pos
                kl = kl_divergence(next_state_prior_a, next_state_posterior_a).sum(dim=1)
                kl_loss += kl.clamp(min=self.args.free_nats).mean()
            kl_loss /= (self.args.chunk_length - 1)

            # compute reconstructed observations and predicted rewards
            flatten_states_a = states_a.view(-1, self.args.stochastic_size)
            flatten_states_pos = states_pos.view(-1, self.args.stochastic_size)
            flatten_rnn_hiddens_a = rnn_hiddens_a.view(-1, self.args.deterministic_size)
            flatten_rnn_hiddens_pos = rnn_hiddens_pos.view(-1, self.args.deterministic_size)
            feature_a = torch.cat([flatten_states_a, flatten_rnn_hiddens_a], dim=1)
            feature_pos = torch.cat([flatten_states_pos, flatten_rnn_hiddens_pos], dim=1)
            predicted_rewards = self.reward_model(flatten_states_a, flatten_rnn_hiddens_a)
            predicted_rewards = predicted_rewards.view(self.args.chunk_length, self.args.batch_size, 1)

            # compute loss for observation and reward
            if self.args.image_loss_type == 'augment_contrast':
                logits, labels = self.curl_model.info_nce_loss(feature_a, feature_pos)
                obs_loss = self.cross_entropy_loss(logits, labels)
            elif self.args.image_loss_type == 'aug_obs_embed_contrast':
                flatten_embeds_pos = embedded_observations_pos.view(-1, self.embed_size)
                logits, labels = self.curl_model.info_nce_loss(feature_a, flatten_embeds_pos)
                obs_loss = self.cross_entropy_loss(logits, labels)
            reward_loss = 0.5 * mse_loss(predicted_rewards[1:], rewards[:-1])
            flatten_states = torch.cat((flatten_states_a, flatten_states_pos))
            flatten_rnn_hiddens = torch.cat((flatten_rnn_hiddens_a, flatten_rnn_hiddens_pos))

        # add all losses and update model parameters with gradient descent
        model_loss = self.args.kl_scale * kl_loss + obs_loss + reward_loss

        # log losses
        if self.model_itr % self.args.model_log_freq == 0:
            self.logger.log('train_model/observation_loss', obs_loss.item(), self.model_itr)
            self.logger.log('train_model/reward_loss', reward_loss.item(), self.model_itr)
            self.logger.log('train_model/kl_loss', kl_loss.item(), self.model_itr)
            self.logger.log('train_model/overall_loss', model_loss.item(), self.model_itr)
        return model_loss, flatten_states, flatten_rnn_hiddens, rewards, actions

    def get_video(self, actions, obs):
        with torch.no_grad():
            observations = np.array(obs)
            observations = torch.as_tensor(observations, device=self.device).transpose(0, 1)
            ground_truth = observations + 0.5
            ground_truth = ground_truth.squeeze(1).unsqueeze(0)

            actions = torch.as_tensor(actions, device=self.device).transpose(0, 1).float()

            # embed observations
            embedded_observation = self.observation_encoder(observations[0].reshape(-1, 3, 64, 64))

            # initialize rnn hidden state
            rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)

            # imagine trajectory
            imagined = []
            state = self.rssm.posterior(rnn_hidden, embedded_observation).sample()
            for action in actions:
                state_prior, rnn_hidden = self.rssm.prior(state, action, rnn_hidden)
                state = state_prior.sample()
                predicted_obs = self.observation_decoder(state, rnn_hidden)
                imagined.append(predicted_obs)
            imagined = torch.stack(imagined).squeeze(1).unsqueeze(0) + 0.5
            video = torch.cat((ground_truth, imagined), dim=0)
        return video
