import os
import time
import torch
import numpy as np
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

from agents.value_agent import PolicyAgent
from algos.dreamer_base import DreamerBase
from models.action_model import ActionModel
from models.value_model import ValueModel
from utils.misc import lambda_target
from utils.sampler import Sampler


class DreamerValue(DreamerBase):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # value model
        self.value_model = ValueModel(self.feature_size)

        # action model
        self.action_model = ActionModel(self.feature_size, self.args.action_dim)

        # gpu settings
        self.value_model.to(self.device)
        self.action_model.to(self.device)

        # action model optimizer
        self.value_model_optimizer = torch.optim.Adam(self.value_model.parameters(),
                                                      lr=self.args.value_lr,
                                                      eps=self.args.value_eps)

        # value model optimizer
        self.action_model_optimizer = torch.optim.Adam(self.action_model.parameters(),
                                                       lr=self.args.action_lr,
                                                       eps=self.args.action_eps)

        self.agent = PolicyAgent(self.device,
                                 self.args.action_dim,
                                 self.rssm,
                                 self.observation_encoder,
                                 self.action_model,
                                 self.args.exploration_noise_var)
        self.sampler = Sampler(env, self.replay_buffer, self.agent)

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
                self.optimize_value(flatten_states, flatten_rnn_hiddens)

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

    def optimize_value(self, flatten_states, flatten_rnn_hiddens):
        action_loss, value_loss = self.get_losses(flatten_states, flatten_rnn_hiddens)
        self.value_model_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        clip_grad_norm_(self.value_model.parameters(), self.args.grad_clip)
        self.value_model_optimizer.step()

        self.action_model_optimizer.zero_grad()
        action_loss.backward()
        clip_grad_norm_(self.action_model.parameters(), self.args.grad_clip)
        self.action_model_optimizer.step()

        if self.args.full_tb_log and self.model_itr % self.args.model_log_freq == 0:
            self.action_model.log(self.logger, self.model_itr)
            self.value_model.log(self.logger, self.model_itr)

    def get_losses(self, flatten_states, flatten_rnn_hiddens):
        # detach gradient because Dreamer doesn't update model with actor-critic loss
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
            actions = self.action_model(flatten_states, flatten_rnn_hiddens)
            flatten_states_prior, flatten_rnn_hiddens = self.rssm.prior(flatten_states, actions, flatten_rnn_hiddens)
            flatten_states = flatten_states_prior.rsample()
            imagined_states[h] = flatten_states
            imagined_rnn_hiddens[h] = flatten_rnn_hiddens

        # compute rewards and values
        flatten_imagined_states = imagined_states.view(-1, self.args.stochastic_size)
        flatten_imagined_rnn_hiddens = imagined_rnn_hiddens.view(-1, self.args.deterministic_size)
        imagined_rewards = self.reward_model(flatten_imagined_states, flatten_imagined_rnn_hiddens)
        imagined_rewards = imagined_rewards.view(self.args.imagination_horizon + 1, -1)
        imagined_values = self.value_model(flatten_imagined_states, flatten_imagined_rnn_hiddens)
        imagined_values = imagined_values.view(self.args.imagination_horizon + 1, -1)

        # compute lambda target
        lambda_target_values = lambda_target(imagined_rewards, imagined_values, self.args.gamma, self.args.lambda_)

        # action loss
        action_loss = -1 * (lambda_target_values.mean())

        # value loss
        value_loss = 0.5 * mse_loss(imagined_values, lambda_target_values.detach())

        # log losses
        if self.model_itr % self.args.model_log_freq == 0:
            self.logger.log('train_action/action_loss', action_loss.item(), self.model_itr)
            self.logger.log('train_value/value_loss', value_loss.item(), self.model_itr)
        return action_loss, value_loss

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
            'action_model_state_dict': self.action_model.state_dict(),
            'value_model_state_dict': self.value_model.state_dict()
        }, os.path.join(model_path, model_name))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.observation_encoder.load_state_dict(checkpoint['observation_encoder_state_dict'])
        self.observation_decoder.load_state_dict(checkpoint['observation_decoder_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.rssm.load_state_dict(checkpoint['rssm_state_dict'])
        self.action_model.load_state_dict(checkpoint['action_model_state_dict'])
        self.value_model.load_state_dict(checkpoint['value_model_state_dict'])
