import os
import time

import numpy as np
import torch

from models.decoder import ObservationDecoder
from models.dense import DenseModel
from models.encoder import ObservationEncoder
from models.rssm import RSSMTransition, RSSMRepresentation, RSSMRollout, get_feat, get_dist, RSSMState, get_state
from utils.misc import get_parameters, infer_leading_dims


class DreamerBase:
    def __init__(self,
                 logger=None,
                 sampler=None,
                 replay_buffer=None,
                 device=None,
                 tensorboard_log_freq=1000,
                 image_shape=None,
                 action_shape=None,
                 reward_shape=None,
                 stochastic_size=200,
                 deterministic_size=30,
                 reward_layers=3,
                 reward_hidden=200,
                 model_lr=6e-4,
                 grad_clip=100.0,
                 free_nats=3,
                 kl_scale=1,
                 action_repeat=1):
        super().__init__()

        self.logger = logger
        self.sampler = sampler
        self.replay_buffer = replay_buffer
        self.device = device
        self.tensorboard_log_freq = tensorboard_log_freq
        self.grad_clip = grad_clip
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.action_repeat = action_repeat
        self.training = True
        self.eval = False
        self.itr = 1
        self.model_itr = 1
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
        self.feature_size = stochastic_size + deterministic_size
        self.reward_model = DenseModel(self.feature_size, reward_shape, reward_layers, reward_hidden)

        # bundle models
        self.model_modules = [self.observation_encoder,
                              self.observation_decoder,
                              self.reward_model,
                              self.transition,
                              self.representation]

        # gpu settings
        self.observation_encoder.to(self.device)
        self.observation_decoder.to(self.device)
        self.reward_model.to(self.device)
        self.transition.to(self.device)
        self.representation.to(self.device)

        # model optimizer
        self.model_optimizer = torch.optim.Adam(
            get_parameters(self.model_modules),
            lr=model_lr
        )

    def optimize_model(self, samples):
        # compute model loss
        model_loss, post = self.model_loss(samples)

        # take gradient step
        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_parameters(self.model_modules), self.grad_clip)
        self.model_optimizer.step()
        return post

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
            if self.model_itr % self.tensorboard_log_freq == 0:
                self.logger.log('train_model/image_loss', image_loss, self.step)
                self.logger.log('train_model/reward_loss', reward_loss, self.step)
                self.logger.log('train_model/transition_loss', div, self.step)
                self.logger.log('train_model/model_loss', model_loss, self.step)
                reconstruction_video = self.get_reconstruction_video(observation, image_pred)
                imagination_video = self.get_imagination_video(observation, image_pred, action, post)
                self.logger.log_video('train_model/reconstruction_video', reconstruction_video, self.step)
                self.logger.log_video('train_model/imagination_video', imagination_video, self.step)
        return model_loss, post

    def get_state_representation(self, observation: torch.Tensor, prev_action: torch.Tensor = None,
                                 prev_state: RSSMState = None):
        # first observation has no prev_state and prev_action
        if prev_action is None:
            prev_action = torch.zeros(self.action_size, device=observation.device, dtype=observation.dtype)
        if prev_state is None:
            prev_state = self.representation.initial_state(1, device=prev_action.device,
                                                           dtype=prev_action.dtype)
        # reshape variables
        observation = torch.unsqueeze(observation, 0)
        prev_action = torch.unsqueeze(prev_action, 0)

        # encode observations
        obs_embed = self.observation_encoder(observation)
        _, state = self.representation(obs_embed, prev_action, prev_state)
        return state

    def evaluate(self, eval_episodes, eval_episode_length, policy, save_eval_video, video_dir, render_eval):
        eval_start_time = time.time()
        self.training = False
        self.eval = True

        # record video
        if save_eval_video:
            self.sampler.reset_video_recorder(video_dir, 'video_eval' + str(self.itr))
        with torch.no_grad():
            episodes, total_steps = self.sampler.collect_policy_episodes(eval_episodes, eval_episode_length, policy,
                                                                         self.get_state_representation,
                                                                         self.device,
                                                                         render=render_eval)

        # calculate episode rewards
        all_ep_rewards = []
        for episode in episodes:
            ep_reward = np.sum(episode.rewards)
            all_ep_rewards.append(ep_reward)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)

        # log results
        eval_time = time.time() - eval_start_time
        self.logger.log('eval/mean_ep_reward', mean_ep_reward, self.step)
        self.logger.log('eval/best_ep_reward', best_ep_reward, self.step)
        self.logger.log('eval/eval_time', eval_time, self.step)

    def get_reconstruction_video(self, observation, image_pred, n_video=2):
        ground_truth = observation[:n_video, :] + 0.5
        reconstruction = image_pred.mean[:n_video, :] + 0.5
        reconstruction_error = (reconstruction - ground_truth + 1) / 2
        reconstruction_video = torch.cat((ground_truth, reconstruction, reconstruction_error), dim=3)
        return reconstruction_video

    def get_imagination_video(self, observation, image_pred, action, post, n_video=2, t_video=20):
        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3)
        ground_truth = observation[:n_video, :] + 0.5
        reconstruction = image_pred.mean[:n_video, :t_video] + 0.5
        prev_state = get_state(post, n_video, t_video - 1)
        prior = self.rollout.rollout_transition(batch_t - t_video, action[:n_video, t_video:], prev_state)
        imagined = self.observation_decoder(get_feat(prior)).mean + 0.5
        model = torch.cat((reconstruction, imagined), dim=1)
        imagined_error = (model - ground_truth + 1) / 2
        imagined_video = torch.cat((ground_truth, model, imagined_error), dim=3)
        return imagined_video
