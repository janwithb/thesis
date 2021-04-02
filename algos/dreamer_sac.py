import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from algos.dreamer_base import DreamerBase
from models.critic import DoubleQCritic
from models.rssm import get_feat
from models.sac_actor import DiagGaussianActor
from utils.misc import flatten_rssm_state, soft_update_params
from utils.replay_buffer import ReplayBuffer


class DreamerSAC(DreamerBase):
    def __init__(self,
                 logger=None,
                 sampler=None,
                 replay_buffer=None,
                 device=None,
                 action_space=None,
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
                 action_repeat=1,
                 representation_loss='contrastive',
                 random_crop_size=64,
                 sac_replay_buffer_capacity=100000,
                 discount=0.99,
                 imagine_horizon=15,
                 learnable_temperature=True,
                 critic_tau=0.005,
                 critic_hidden=1024,
                 critic_layers=2,
                 critic_lr=1e-4,
                 critic_betas=None,
                 critic_target_update_frequency=2,
                 actor_update_frequency=1,
                 actor_hidden=1024,
                 actor_layers=2,
                 actor_lr=1e-4,
                 actor_betas=None,
                 log_std_bounds=None,
                 init_temperature=0.1,
                 alpha_lr=1e-4,
                 alpha_betas=None):

        super().__init__(logger=logger,
                         sampler=sampler,
                         replay_buffer=replay_buffer,
                         device=device,
                         tensorboard_log_freq=tensorboard_log_freq,
                         image_shape=image_shape,
                         action_shape=action_shape,
                         reward_shape=reward_shape,
                         stochastic_size=stochastic_size,
                         deterministic_size=deterministic_size,
                         reward_layers=reward_layers,
                         reward_hidden=reward_hidden,
                         model_lr=model_lr,
                         grad_clip=grad_clip,
                         free_nats=free_nats,
                         kl_scale=kl_scale,
                         action_repeat=action_repeat,
                         representation_loss=representation_loss,
                         random_crop_size=random_crop_size)

        self.sac_replay_buffer = ReplayBuffer(
            self.feature_size,
            self.action_size,
            sac_replay_buffer_capacity,
            self.device
        )

        self.discount = discount
        self.imagine_horizon = imagine_horizon
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature
        self.sac_itr = 0

        # critic model
        self.critic = DoubleQCritic(self.feature_size, self.action_size, critic_hidden, critic_layers)

        # actor target model
        self.critic_target = DoubleQCritic(self.feature_size, self.action_size, critic_hidden, critic_layers)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # actor model
        self.actor = DiagGaussianActor(self.feature_size, self.action_size, actor_hidden, actor_layers, log_std_bounds)

        # log_alpha model
        self.log_alpha = torch.tensor(np.log(init_temperature))
        self.log_alpha.requires_grad = True

        # gpu settings
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        self.actor.to(self.device)
        self.log_alpha.to(self.device)

        # set target entropy
        self.target_entropy = -self.action_size

        # define action range
        self.action_range = [
            float(action_space.low.min()),
            float(action_space.high.max())
        ]

        # actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
            betas=actor_betas
        )

        # critic optimizer
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            betas=critic_betas
        )

        # log_alpha optimizer
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_lr,
            betas=alpha_betas
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self,
              init_episodes=100,
              init_episode_length=100,
              iter_episodes=10,
              iter_episode_length=100,
              training_iterations=100,
              model_iterations=100,
              sac_iterations=100,
              batch_size=64,
              chunk_size=50,
              sac_batch_size=64,
              render_training=False,
              save_iter_model=False,
              save_iter_model_freq=10,
              model_dir=None,
              eval_freq=1,
              eval_episodes=5,
              eval_episode_length=100,
              render_eval=False,
              save_eval_video=False,
              video_dir=None):

        # collect initial episodes
        episodes, total_steps = self.sampler.collect_random_episodes(init_episodes, init_episode_length,
                                                                     render=render_training)
        self.replay_buffer.add(episodes)
        self.step += total_steps * self.action_repeat

        # main training loop
        for it in tqdm(range(training_iterations), desc='Training progress'):
            itr_start_time = time.time()
            self.set_mode('train')

            # model training loop
            for _ in tqdm(range(model_iterations), desc='Model Training'):
                samples = self.replay_buffer.get_chunk_batch(batch_size, chunk_size)
                post = self.optimize_model(samples)
                self.update_sac_replay_buffer(post)
                self.model_itr += 1

            # SAC training loop
            for _ in tqdm(range(sac_iterations), desc='SAC Training'):
                self.optimize_sac(self.sac_replay_buffer, sac_batch_size)
                self.sac_itr += 1

            # collect new data
            episodes, total_steps = self.sampler.collect_policy_episodes(iter_episodes, iter_episode_length,
                                                                         self.exploration_policy,
                                                                         self.get_state_representation,
                                                                         self.device,
                                                                         render=render_training)

            self.replay_buffer.add(episodes)
            self.step += total_steps * self.action_repeat

            # save model frequently
            if save_iter_model and it % save_iter_model_freq == 0:
                self.save_model(model_dir, 'model_iter_' + str(it))

            itr_time = time.time() - itr_start_time
            self.logger.log('train/itr_time', itr_time, self.step)
            self.itr += 1

            # evaluate policy
            if it % eval_freq == 0:
                self.evaluate(eval_episodes, eval_episode_length, self.exploration_policy,
                              save_eval_video, video_dir, render_eval)

    def update_sac_replay_buffer(self, post):
        # remove gradients from tensors
        with torch.no_grad():
            flat_post = flatten_rssm_state(post)
            flat_post_feat = get_feat(flat_post)

            # imagine trajectories
            imag_dist, actions = self.rollout.rollout_policy(self.imagine_horizon, self.policy, flat_post)

            # Use state features (deterministic and stochastic) to predict the image and reward
            imag_feat = get_feat(imag_dist)  # [horizon, batch_t * batch_b, feature_size]

            # freeze model parameters as only action model gradients needed
            imag_reward = self.reward_model(imag_feat).mean

        # add imagined transitions to replay buffer
        for i in range(len(imag_feat)):
            state_batch = imag_feat[i]
            reward_batch = imag_reward[i]
            actions_batch = actions[i]
            obs = flat_post_feat[i]
            for j in range(len(state_batch)):
                next_obs = state_batch[j]
                reward = reward_batch[j]
                action = actions_batch[j]
                self.sac_replay_buffer.add(obs.detach(), action.detach(), reward.detach(), next_obs.detach(), False, False)
                obs = next_obs

    def optimize_sac(self, replay_buffer, sac_batch_size):
        # sample transition
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(sac_batch_size)

        # update critic
        self.optimize_critic(obs, action, reward, next_obs, not_done_no_max)

        # update actor
        if self.sac_itr % self.actor_update_frequency == 0:
            log_prob = self.optimize_actor(obs)
            if self.learnable_temperature:
                self.optimize_alpha(log_prob)

        # soft update critic target
        if self.sac_itr % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target,
                               self.critic_tau)

    def optimize_critic(self, obs, action, reward, next_obs, not_done):
        # compute critic loss
        critic_loss = self.critic_loss(obs, action, reward, next_obs, not_done)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.sac_itr % self.tensorboard_log_freq == 0:
            self.critic.log(self.logger, self.step)

    def optimize_actor(self, obs):
        # compute critic loss
        actor_loss, log_prob = self.actor_loss(obs)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.sac_itr % self.tensorboard_log_freq == 0:
            self.actor.log(self.logger, self.step)
        return log_prob

    def optimize_alpha(self, log_prob):
        # compute alpha loss
        alpha_loss = self.alpha_loss(log_prob)

        # optimize the alpha
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def critic_loss(self, obs, action, reward, next_obs, not_done):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # log losses
        if self.sac_itr % self.tensorboard_log_freq == 0:
            self.logger.log('train_critic/critic_loss', critic_loss, self.step)
        return critic_loss

    def actor_loss(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # log losses
        if self.sac_itr % self.tensorboard_log_freq == 0:
            self.logger.log('train_actor/actor_loss', actor_loss, self.step)
            self.logger.log('train_actor/target_entropy', self.target_entropy, self.step)
            self.logger.log('train_actor/entropy', -log_prob.mean(), self.step)
        return actor_loss, log_prob

    def alpha_loss(self, log_prob):
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        # log losses
        if self.sac_itr % self.tensorboard_log_freq == 0:
            self.logger.log('train_alpha/value_loss', alpha_loss, self.step)
            self.logger.log('train_alpha/value', self.alpha, self.step)
        return alpha_loss

    def policy(self, state, sample=False):
        feat = get_feat(state)
        dist = self.actor(feat)
        action = dist.sample() if sample else dist.mode
        action = action.clamp(*self.action_range)
        return action, dist

    def exploration_policy(self, state):
        if self.training:
            action, _ = self.policy(state, sample=True)
        elif self.eval:
            action, _ = self.policy(state)
        action = np.squeeze(action.detach().cpu().numpy(), axis=0)
        return action

    def save_model(self, model_path, model_name):
        torch.save({
            'observation_encoder_state_dict': self.observation_encoder.state_dict(),
            'observation_decoder_state_dict': self.observation_decoder.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'representation_state_dict': self.representation.state_dict(),
            'transition_state_dict': self.transition.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'log_alpha_state_dict': self.log_alpha.state_dict()
        }, os.path.join(model_path, model_name))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.observation_encoder.load_state_dict(checkpoint['observation_encoder_state_dict'])
        self.observation_decoder.load_state_dict(checkpoint['observation_decoder_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.representation.load_state_dict(checkpoint['representation_state_dict'])
        self.transition.load_state_dict(checkpoint['transition_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.log_alpha.load_state_dict(checkpoint['log_alpha_state_dict'])
