import os
import time
import torch
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F

from agents.policy_agent import PolicyAgent
from algos.dreamer_base import DreamerBase
from models.critic_model import DoubleQCritic
from models.sac_actor_model import DiagGaussianActor
from utils.misc import soft_update_params
from utils.replay_buffer import ReplayBuffer
from utils.sampler import Sampler


class DreamerSAC(DreamerBase):
    def __init__(self, env, logger, replay_buffer, device, args):
        super().__init__(logger, replay_buffer, device, args)
        self.args = args
        self.sac_itr = 0

        self.sac_replay_buffer = ReplayBuffer(
            args.feature_size,
            args.action_dim,
            args.sac_replay_buffer_capacity,
            self.device
        )

        # critic model
        self.critic = DoubleQCritic(self.feature_size,
                                    args.action_dim,
                                    args.critic_hidden,
                                    args.critic_layers)

        # actor target model
        self.critic_target = DoubleQCritic(self.feature_size,
                                           args.action_dim,
                                           args.critic_hidden,
                                           args.critic_layers)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # actor model
        self.actor = DiagGaussianActor(self.feature_size,
                                       args.action_dim,
                                       args.actor_hidden,
                                       args.actor_layers,
                                       args.log_std_bounds)

        # log_alpha model
        self.log_alpha = torch.tensor(np.log(args.init_temperature))
        self.log_alpha.requires_grad = True

        # gpu settings
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        self.actor.to(self.device)
        self.log_alpha.to(self.device)

        # set target entropy
        self.target_entropy = -args.action_dim

        # actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=args.actor_lr,
            betas=args.actor_betas
        )

        # critic optimizer
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=args.critic_lr,
            betas=args.critic_betas
        )

        # log_alpha optimizer
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=args.alpha_lr,
            betas=args.alpha_betas
        )

        self.agent = PolicyAgent(self.device, self.rssm, self.observation_encoder, self.action_model)
        self.sampler = Sampler(env, replay_buffer, self.agent)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self):
        episode = 0

        # collect initial episodes
        _, _, _, all_episode_rewards = self.sampler.collect_episodes(self.args.init_episodes,
                                                                     random=True,
                                                                     render=self.args.render_training)
        episode += self.args.init_episodes
        mean_ep_reward = np.mean(all_episode_rewards)
        best_ep_reward = np.max(all_episode_rewards)
        self.logger.log('train/mean_ep_reward', mean_ep_reward, episode)
        self.logger.log('train/best_ep_reward', best_ep_reward, episode)

        # main training loop
        for it in tqdm(range(self.args.training_iterations), desc='Training progress'):
            itr_start_time = time.time()

            # collect experience on policy
            _, _, _, all_episode_rewards = self.sampler.collect_episodes(self.args.agent_episodes,
                                                                         exploration=True,
                                                                         render=self.args.render_training)
            episode += self.args.agent_episodes
            mean_ep_reward = np.mean(all_episode_rewards)
            best_ep_reward = np.max(all_episode_rewards)
            self.logger.log('train/mean_ep_reward', mean_ep_reward, episode)
            self.logger.log('train/best_ep_reward', best_ep_reward, episode)

            # model training loop
            for _ in range(self.args.model_iterations):
                flatten_states, flatten_rnn_hiddens = self.optimize_model()
                self.update_sac_replay_buffer(flatten_states, flatten_rnn_hiddens)

            # model training loop
            for _ in range(self.args.sac_iterations):
                self.optimize_sac(self.sac_replay_buffer, args.sac_batch_size)

            # save model frequently
            if self.args.save_iter_model and episode % self.args.save_iter_model_freq == 0:
                self.save_model(episode, 'model_iter_' + str(episode))

            itr_time = time.time() - itr_start_time
            self.logger.log('train/itr_time', itr_time, episode)

            # evaluate policy
            if it % self.args.eval_freq == 0:
                eval_start_time = time.time()

                # collect experience on policy
                actions, obs, _, all_episode_rewards = self.sampler.collect_episodes(self.args.eval_episodes,
                                                                                     render=self.args.render_training)
                mean_ep_reward = np.mean(all_episode_rewards)
                best_ep_reward = np.max(all_episode_rewards)
                video = self.get_video(actions, obs)

                # log results
                eval_time = time.time() - eval_start_time
                self.logger.log('eval/mean_ep_reward', mean_ep_reward, episode)
                self.logger.log('eval/best_ep_reward', best_ep_reward, episode)
                self.logger.log('eval/eval_time', eval_time, episode)
                self.logger.log_video('eval/ep_video', video, episode)

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
                self.sac_replay_buffer.add(obs.detach(), action.detach(), reward.detach(), next_obs.detach(), False,
                                           False)
                obs = next_obs

    def optimize_sac(self, replay_buffer, sac_batch_size):
        # sample transition
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(sac_batch_size)

        # update critic
        self.optimize_critic(obs, action, reward, next_obs, not_done_no_max)

        # update actor
        if self.sac_itr % self.args.actor_update_frequency == 0:
            log_prob = self.optimize_actor(obs)
            if self.args.learnable_temperature:
                self.optimize_alpha(log_prob)

        # soft update critic target
        if self.sac_itr % self.args.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.args.critic_tau)
        self.sac_itr += 1

    def optimize_critic(self, obs, action, reward, next_obs, not_done):
        # compute critic loss
        critic_loss = self.critic_loss(obs, action, reward, next_obs, not_done)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.args.full_tb_log and self.sac_itr % self.args.tensorboard_log_freq == 0:
            self.critic.log(self.logger, self.sac_itr)

    def optimize_actor(self, obs):
        # compute critic loss
        actor_loss, log_prob = self.actor_loss(obs)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.full_tb_log and self.sac_itr % self.args.tensorboard_log_freq == 0:
            self.actor.log(self.logger, self.sac_itr)
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
        target_Q = reward + (not_done * self.args.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # log losses
        if self.sac_itr % self.args.tensorboard_log_freq == 0:
            self.logger.log('train_critic/critic_loss', critic_loss, self.sac_itr)
        return critic_loss

    def actor_loss(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # log losses
        if self.sac_itr % self.args.tensorboard_log_freq == 0:
            self.logger.log('train_actor/actor_loss', actor_loss, self.sac_itr)
            self.logger.log('train_actor/target_entropy', self.target_entropy, self.sac_itr)
            self.logger.log('train_actor/entropy', -log_prob.mean(), self.sac_itr)
        return actor_loss, log_prob

    def alpha_loss(self, log_prob):
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        # log losses
        if self.sac_itr % self.args.tensorboard_log_freq == 0:
            self.logger.log('train_alpha/value_loss', alpha_loss, self.sac_itr)
            self.logger.log('train_alpha/value', self.alpha, self.sac_itr)
        return alpha_loss

    def get_video(self, actions, obs):
        with torch.no_grad():
            observations = torch.as_tensor(obs, device=self.device).transpose(0, 1)
            ground_truth = observations + 0.5
            ground_truth = ground_truth.squeeze(1).unsqueeze(0)
            actions = torch.as_tensor(actions, device=self.device).transpose(0, 1)

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

    def save_model(self, model_path, model_name):
        torch.save({
            'observation_encoder_state_dict': self.observation_encoder.state_dict(),
            'observation_decoder_state_dict': self.observation_decoder.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'rssm_state_dict': self.rssm.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'log_alpha': self.log_alpha
        }, os.path.join(model_path, model_name))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.observation_encoder.load_state_dict(checkpoint['observation_encoder_state_dict'])
        self.observation_decoder.load_state_dict(checkpoint['observation_decoder_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.rssm.load_state_dict(checkpoint['rssm_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.log_alpha = checkpoint['log_alpha_state_dict']

