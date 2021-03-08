import os

import numpy as np
import torch

from models.actor import ActorModel
from models.decoder import ObservationDecoder
from models.dense import DenseModel
from models.encoder import ObservationEncoder
from models.rssm import RSSMTransition, RSSMRepresentation, RSSMRollout, get_feat, get_dist, RSSMState
from utils.logger import Logger
from utils.misc import get_parameters, infer_leading_dims, FreezeParameters, compute_return, flatten_rssm_state


class Dreamer:
    def __init__(self,
                 logger: Logger,
                 device,
                 tensorboard_log_freq: int,
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
                 grad_clip: float,
                 free_nats: int,
                 kl_scale: int,
                 value_shape: tuple,
                 value_layers: int,
                 value_hidden: int,
                 actor_layers: int,
                 actor_hidden: int,
                 actor_dist: str,
                 imagine_horizon: int,
                 discount: float,
                 discount_lambda: float,
                 train_noise: float,
                 eval_noise: float,
                 expl_method: str,
                 expl_amount: float,
                 expl_decay: float,
                 expl_min: float):
        super().__init__()

        self.logger = logger
        self.device = device
        self.tensorboard_log_freq = tensorboard_log_freq
        self.grad_clip = grad_clip
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.imagine_horizon = imagine_horizon
        self.discount = discount
        self.discount_lambda = discount_lambda
        self.actor_dist = actor_dist
        self.train_noise = train_noise
        self.eval_noise = eval_noise
        self.expl_method = expl_method
        self.expl_amount = expl_amount
        self.expl_decay = expl_decay
        self.expl_min = expl_min
        self.training = True
        self.eval = False
        self.itr = 1
        self.img_itr = 1
        self.step = 0

        # encoder model
        self.observation_encoder = ObservationEncoder(shape=image_shape)
        encoder_embed_size = self.observation_encoder.embed_size
        decoder_embed_size = stochastic_size + deterministic_size

        # decoder model
        self.observation_decoder = ObservationDecoder(embed_size=decoder_embed_size, shape=image_shape)
        self.action_size = np.prod(action_shape)

        # recurrent state space model
        self.transition = RSSMTransition(self.action_size)
        self.representation = RSSMRepresentation(self.transition, encoder_embed_size, self.action_size)
        self.rollout = RSSMRollout(self.representation, self.transition)

        # reward model
        feature_size = stochastic_size + deterministic_size
        self.reward_model = DenseModel(feature_size, reward_shape, reward_layers, reward_hidden)

        # actor model
        self.actor_model = ActorModel(self.action_size, feature_size, actor_hidden, actor_layers, actor_dist)

        # value model
        self.value_model = DenseModel(feature_size, value_shape, value_layers, value_hidden)

        # bundle models
        self.model_modules = [self.observation_encoder,
                              self.observation_decoder,
                              self.reward_model,
                              self.transition,
                              self.representation]
        self.actor_modules = [self.actor_model]
        self.value_modules = [self.value_model]

        # gpu settings
        self.observation_encoder.to(self.device)
        self.observation_decoder.to(self.device)
        self.reward_model.to(self.device)
        self.transition.to(self.device)
        self.representation.to(self.device)
        self.actor_model.to(self.device)
        self.value_model.to(self.device)

        # model optimizer
        self.model_optimizer = torch.optim.Adam(
            get_parameters(self.model_modules),
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

    def optimize(self, samples):
        post = self.optimize_model(samples)
        imag_feat, discount, returns = self.optimize_actor(post)
        self.optimize_value(imag_feat, discount, returns)

    def optimize_model(self, samples):
        # compute model loss
        model_loss, post = self.model_loss(samples)

        # take gradient step
        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_parameters(self.model_modules), self.grad_clip)
        self.model_optimizer.step()
        return post

    def optimize_actor(self, post):
        # compute actor loss
        actor_loss, imag_feat, discount, returns = self.actor_loss(post)

        # take gradient step
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_modules), self.grad_clip)
        self.actor_optimizer.step()
        return imag_feat, discount, returns

    def optimize_value(self, imag_feat, discount, returns):
        # compute model loss
        value_loss = self.value_loss(imag_feat, discount, returns)

        # take gradient step
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_parameters(self.value_modules), self.grad_clip)
        self.value_optimizer.step()

    def model_loss(self, samples):
        # convert samples to tensors
        observation = torch.as_tensor(np.array([chunk.states for chunk in samples]), device=self.device)
        action = torch.as_tensor(np.array([chunk.actions for chunk in samples]), device=self.device)
        reward = torch.as_tensor(np.array([chunk.rewards for chunk in samples]), device=self.device)
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

        # log losses
        with torch.no_grad():
            if self.img_itr % self.tensorboard_log_freq == 0:
                self.logger.log('train_model/image_loss', image_loss, self.step)
                self.logger.log('train_model/reward_loss', reward_loss, self.step)
                self.logger.log('train_model/transition_loss', div, self.step)
                self.logger.log('train_model/model_loss', model_loss, self.step)
                reconstruction_video = self.get_reconstruction_video(observation, image_pred)
                imagination_video = self.get_imagination_video(observation, action)
                self.logger.log_video('train_model/reconstruction_video', reconstruction_video, self.step)
                self.logger.log_video('train_model/imagination_video', imagination_video, self.step)
        return model_loss, post

    def actor_loss(self, post):
        # remove gradients from previously calculated tensors
        with torch.no_grad():
            flat_post = flatten_rssm_state(post)

        # imagine trajectories
        with FreezeParameters(self.model_modules):
            imag_dist, _ = self.rollout.rollout_policy(self.imagine_horizon, self.policy, flat_post)

        # Use state features (deterministic and stochastic) to predict the image and reward
        imag_feat = get_feat(imag_dist)  # [horizon, batch_t * batch_b, feature_size]

        # freeze model parameters as only action model gradients needed
        with FreezeParameters(self.model_modules + self.value_modules):
            imag_reward = self.reward_model(imag_feat).mean
            value = self.value_model(imag_feat).mean

        # compute the exponential discounted sum of rewards
        discount_arr = self.discount * torch.ones_like(imag_reward)
        returns = compute_return(imag_reward[:, :-1], value[:, :-1], discount_arr[:, :-1],
                                 bootstrap=value[:, -1], lambda_=self.discount_lambda)

        # make the top row 1 so the cumulative product starts with discount^0
        discount_arr = torch.cat([torch.ones_like(discount_arr[:, :1]), discount_arr[:, 1:]], 1)
        discount = torch.cumprod(discount_arr[:, :-1], 1)
        actor_loss = -torch.mean(discount * returns)

        # log loss
        if self.img_itr % self.tensorboard_log_freq == 0:
            self.logger.log('train_actor/actor_loss', actor_loss, self.step)
        return actor_loss, imag_feat, discount, returns

    def value_loss(self, imag_feat, discount, returns):
        # remove gradients from previously calculated tensors
        with torch.no_grad():
            value_feat = imag_feat[:, :-1].detach()
            value_discount = discount.detach()
            value_target = returns.detach()
        value_pred = self.value_model(value_feat)
        log_prob = value_pred.log_prob(value_target)
        value_loss = -torch.mean(value_discount * log_prob.unsqueeze(2))

        # log loss
        if self.img_itr % self.tensorboard_log_freq == 0:
            self.logger.log('train_value/value_loss', value_loss, self.step)
        return value_loss

    def policy(self, state: RSSMState):
        feat = get_feat(state)
        action_dist = self.actor_model(feat)
        if self.actor_dist == 'tanh_normal':
            if self.training:
                action = action_dist.rsample()
            else:
                action = action_dist.mode()
        elif self.actor_dist == 'one_hot':
            action = action_dist.sample()
            # this doesn't change the value, but gives us straight-through gradients
            action = action + action_dist.probs - action_dist.probs.detach()
        elif self.actor_dist == 'relaxed_one_hot':
            action = action_dist.rsample()
        else:
            action = action_dist.sample()
        return action, action_dist

    def exploration_policy(self, state):
        action, _ = self.policy(state)
        action = self.exploration(action)
        action = np.squeeze(action.detach().cpu().numpy(), axis=0)
        return action

    def exploration(self, action: torch.Tensor) -> torch.Tensor:
        if self.training:
            expl_amount = self.train_noise
            # linear decay
            if self.expl_decay:
                expl_amount = expl_amount - self.itr / self.expl_decay
            if self.expl_min:
                expl_amount = max(self.expl_min, expl_amount)
        elif self.eval:
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError

        # for continuous actions
        if self.expl_method == 'additive_gaussian':
            noise = torch.randn(*action.shape, device=action.device) * expl_amount
            return torch.clamp(action + noise, -1, 1)
        if self.expl_method == 'completely_random':  # For continuous actions
            if expl_amount == 0:
                return action
            else:
                # scale to [-1, 1]
                return torch.rand(*action.shape, device=action.device) * 2 - 1

        # for discrete actions
        if self.expl_method == 'epsilon_greedy':
            action_dim = action.shape[0]
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, action_dim, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[..., index] = 1
            return action
        raise NotImplementedError(self.expl_method)

    def get_state_representation(self, observation: torch.Tensor, prev_action: torch.Tensor = None,
                                 prev_state: RSSMState = None):
        # first observation has no prev_state and prev_action
        if prev_action is None:
            prev_action = torch.zeros(self.action_size, device=observation.device, dtype=observation.dtype)
        if prev_state is None:
            prev_state = self.representation.initial_state(prev_action.size(0), device=prev_action.device,
                                                           dtype=prev_action.dtype)
        # reshape variables
        observation = torch.unsqueeze(observation, 0)
        prev_action = torch.unsqueeze(prev_action, 0)

        # encode observations
        obs_embed = self.observation_encoder(observation)
        _, state = self.representation(obs_embed, prev_action, prev_state)
        return state

    def get_reconstruction_video(self, observation, image_pred, n_video=2, t_video=20):
        ground_truth = observation[:n_video, :t_video] + 0.5
        reconstruction = image_pred.mean[:n_video, :t_video] + 0.5
        reconstruction_error = (reconstruction - ground_truth + 1) / 2
        reconstruction_video = torch.cat((ground_truth, reconstruction, reconstruction_error), dim=3)
        return reconstruction_video

    def get_imagination_video(self, observation, action, n_video=2, t_video=20):
        ground_truth = observation[:n_video, :t_video] + 0.5
        prev_state = self.representation.initial_state(n_video, device=action.device, dtype=action.dtype)
        prior = self.rollout.rollout_transition(t_video, action[:n_video, :], prev_state)
        imagined = self.observation_decoder(get_feat(prior)).mean + 0.5
        imagined_error = (imagined - ground_truth + 1) / 2
        imagined_video = torch.cat((ground_truth, imagined, imagined_error), dim=3)
        return imagined_video

    def evaluate(self, episodes):
        all_ep_rewards = []
        for episode in episodes:
            ep_reward = np.array(episode.rewards)
            all_ep_rewards.append(ep_reward)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        self.logger.log('eval/mean_ep_reward', mean_ep_reward, self.step)
        self.logger.log('eval/best_ep_reward', best_ep_reward, self.step)

    def save_model(self, model_path, model_name):
        torch.save({
            'observation_encoder_state_dict': self.observation_encoder.state_dict(),
            'observation_decoder_state_dict': self.observation_decoder.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'representation_state_dict': self.representation.state_dict(),
            'transition_state_dict': self.transition.state_dict(),
            'actor_model_state_dict': self.actor_model.state_dict(),
            'value_model_state_dict': self.value_model.state_dict()
        }, os.path.join(model_path, model_name))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.observation_encoder.load_state_dict(checkpoint['observation_encoder_state_dict'])
        self.observation_decoder.load_state_dict(checkpoint['observation_decoder_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.representation.load_state_dict(checkpoint['representation_state_dict'])
        self.transition.load_state_dict(checkpoint['transition_state_dict'])
        self.actor_model.load_state_dict(checkpoint['actor_model_state_dict'])
        self.value_model.load_state_dict(checkpoint['value_model_state_dict'])
