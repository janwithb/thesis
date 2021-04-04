import os
import time
import torch
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F

from agents.sac_agent import SACAgent
from algos.dreamer_base import DreamerBase
from models.critic_model import DoubleQCritic
from models.sac_actor_model import DiagGaussianActor
from utils.misc import soft_update_params
from utils.replay_buffer import ReplayBuffer
from utils.sampler import Sampler


class DreamerSAC(DreamerBase):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sac_itr = 0
        self.log_std_bounds = [self.args.log_std_min, self.args.log_std_max]
        self.actor_betas = (self.args.actor_beta_min, self.args.actor_beta_max)
        self.critic_betas = (self.args.critic_beta_min, self.args.critic_beta_max)
        self.alpha_betas = (self.args.alpha_beta_min, self.args.alpha_beta_max)

        self.sac_replay_buffer = ReplayBuffer(
            self.feature_size,
            self.args.action_dim,
            self.args.sac_replay_buffer_capacity,
            self.device
        )

        # critic model
        self.critic = DoubleQCritic(self.feature_size,
                                    self.args.action_dim,
                                    self.args.critic_hidden_dim,
                                    self.args.critic_hidden_depth)

        # actor target model
        self.critic_target = DoubleQCritic(self.feature_size,
                                           self.args.action_dim,
                                           self.args.critic_hidden_dim,
                                           self.args.critic_hidden_depth)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # actor model
        self.actor = DiagGaussianActor(self.feature_size,
                                       self.args.action_dim,
                                       self.args.actor_hidden_dim,
                                       self.args.actor_hidden_depth,
                                       self.log_std_bounds)

        # log_alpha model
        self.log_alpha = torch.tensor(np.log(self.args.init_temperature))
        self.log_alpha.requires_grad = True

        # gpu settings
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        self.actor.to(self.device)
        self.log_alpha.to(self.device)

        # set target entropy
        self.target_entropy = -self.args.action_dim

        # actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.args.actor_lr,
            betas=self.actor_betas
        )

        # critic optimizer
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.args.critic_lr,
            betas=self.critic_betas
        )

        # log_alpha optimizer
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.args.alpha_lr,
            betas=self.alpha_betas
        )

        self.agent = SACAgent(self.device, self.args.action_range, self.rssm, self.observation_encoder, self.actor)
        self.sampler = Sampler(env, self.replay_buffer, self.agent)

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
                self.optimize_sac(self.sac_replay_buffer, self.args.sac_batch_size)

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

    def update_sac_replay_buffer(self, flatten_states, flatten_rnn_hiddens):
        with torch.no_grad():
            flatten_states = flatten_states.detach()
            flatten_rnn_hiddens = flatten_rnn_hiddens.detach()

            # prepare tensor to maintain imagined trajectory's states and rnn_hiddens
            imagined_states = torch.zeros(self.args.imagination_horizon + 1,
                                          *flatten_states.shape,
                                          device=flatten_states.device)
            imagined_rnn_hiddens = torch.zeros(self.args.imagination_horizon + 1,
                                               *flatten_rnn_hiddens.shape,
                                               device=flatten_rnn_hiddens.device)
            imagined_states[0] = flatten_states
            imagined_rnn_hiddens[0] = flatten_rnn_hiddens

            # compute imagined trajectory using action from action_model
            for h in range(1, self.args.imagination_horizon + 1):
                features = torch.cat([flatten_states, flatten_rnn_hiddens], dim=1)
                dist = self.actor(features)
                actions = dist.sample()
                actions = actions.clamp(*self.args.action_range)
                flatten_states_prior, flatten_rnn_hiddens = self.rssm.prior(flatten_states, actions,
                                                                            flatten_rnn_hiddens)
                flatten_states = flatten_states_prior.rsample()
                imagined_rewards = self.reward_model(flatten_states, flatten_rnn_hiddens)
                next_features = torch.cat([flatten_states, flatten_rnn_hiddens], dim=1)
                done = torch.zeros(imagined_rewards.shape, dtype=torch.bool)
                done_no_max = torch.zeros(imagined_rewards.shape, dtype=torch.bool)
                self.sac_replay_buffer.add_batch(features, actions, imagined_rewards, next_features, done, done_no_max)

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

        if self.args.full_tb_log and self.sac_itr % self.args.model_log_freq == 0:
            self.critic.log(self.logger, self.sac_itr)

    def optimize_actor(self, obs):
        # compute critic loss
        actor_loss, log_prob = self.actor_loss(obs)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.full_tb_log and self.sac_itr % self.args.model_log_freq == 0:
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
        if self.sac_itr % self.args.model_log_freq == 0:
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
        if self.sac_itr % self.args.model_log_freq == 0:
            self.logger.log('train_actor/actor_loss', actor_loss, self.sac_itr)
            self.logger.log('train_actor/target_entropy', self.target_entropy, self.sac_itr)
            self.logger.log('train_actor/entropy', -log_prob.mean(), self.sac_itr)
        return actor_loss, log_prob

    def alpha_loss(self, log_prob):
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        # log losses
        if self.sac_itr % self.args.model_log_freq == 0:
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

