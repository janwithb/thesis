import numpy as np
import torch

from models.decoder import ObservationDecoder
from models.dense import DenseModel
from models.encoder import ObservationEncoder
from models.rssm import RSSMTransition, RSSMRepresentation, RSSMRollout, get_feat, get_dist
from utils.misc import get_parameters, infer_leading_dims


class Dreamer:
    def __init__(self,
                 image_shape: tuple,
                 action_shape: tuple,
                 reward_shape: tuple,
                 stochastic_size: int,
                 deterministic_size: int,
                 reward_layers: int,
                 reward_hidden: int,
                 model_lr: float,
                 actor_lr: float,
                 value_lr: float,
                 free_nats: int,
                 kl_scale: int,
                 value_shape: tuple,
                 value_layers: int,
                 value_hidden: int):
        super().__init__()

        self.free_nats = free_nats
        self.kl_scale = kl_scale

        # encoder model
        self.observation_encoder = ObservationEncoder(shape=image_shape)
        encoder_embed_size = self.observation_encoder.embed_size
        decoder_embed_size = stochastic_size + deterministic_size

        # decoder model
        self.observation_decoder = ObservationDecoder(embed_size=decoder_embed_size, shape=image_shape)
        action_size = np.prod(action_shape)

        # recurrent state space model
        self.transition = RSSMTransition(action_size)
        self.representation = RSSMRepresentation(self.transition, encoder_embed_size, action_size)
        self.rollout = RSSMRollout(self.representation, self.transition)

        # reward model
        feature_size = stochastic_size + deterministic_size
        self.reward_model = DenseModel(feature_size, reward_shape, reward_layers, reward_hidden)

        # actor model

        # value model
        self.value_model = DenseModel(feature_size, value_shape, value_layers, value_hidden)

        # bundle models
        model_modules = [self.observation_encoder,
                         self.observation_decoder,
                         self.reward_model,
                         self.representation,
                         self.transition]
        self.actor_modules = [self.actor_model]
        self.value_modules = [self.value_model]

        # model optimizer
        self.model_optimizer = torch.optim.Adam(
            get_parameters(model_modules),
            lr=model_lr
        )

        # actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            get_parameters(self.actor_modules),
            lr=actor_lr
        )

        # value optimizer
        self.value_optimizer = torch.optim.Adam(
            get_parameters(self.value_modules),
            lr=value_lr
        )

    def optimize_model(self, samples):
        # compute model loss
        model_loss = self.model_loss(samples)

        # take gradient step
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

    def optimize_actor(self):
        # compute actor loss
        actor_loss = self.actor_loss()

        # take gradient step
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def optimize_value(self):
        # compute model loss
        value_loss = self.value_loss()

        # take gradient step
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def model_loss(self, samples):
        # convert samples to tensors
        observation = torch.as_tensor(np.array([chunk.states for chunk in samples]))
        action = torch.as_tensor(np.array([chunk.actions for chunk in samples]))
        reward = torch.as_tensor(np.array([chunk.rewards for chunk in samples]))
        reward = reward.unsqueeze(2)

        # get dimensions
        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3)

        # encode observations
        embed = self.observation_encoder(observation)

        # rollout model with sample actions
        prev_state = self.representation.initial_state(batch_b, device=action.device, dtype=action.dtype)
        prior, post = self.rollout.rollout_representation(batch_t, embed, action, prev_state)
        feat = get_feat(post)

        # reconstruction loss
        image_pred = self.observation_decoder(feat)
        image_loss = -torch.mean(image_pred.log_prob(observation))

        # reward loss
        reward_pred = self.reward_model(feat)
        reward_loss = -torch.mean(reward_pred.log_prob(reward))

        # transition loss
        prior_dist = get_dist(prior)
        post_dist = get_dist(post)
        div = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        div = torch.max(div, div.new_full(div.size(), self.free_nats))

        # total loss
        model_loss = self.kl_scale * div + reward_loss + image_loss

        return model_loss

    def actor_loss(self):
        actor_loss = 1
        return actor_loss

    def value_loss(self):
        value_loss = 1
        return value_loss
