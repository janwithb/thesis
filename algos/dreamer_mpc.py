import os
import time
import torch
import numpy as np

from tqdm import tqdm
from algos.dreamer_base import DreamerBase
from agents.cem import CEM
from agents.random_shooting import RandomShooting
from utils.sampler import Sampler


class DreamerMPC(DreamerBase):
    def __init__(self, env, logger, replay_buffer, device, args):
        super().__init__(logger, replay_buffer, device, args)

        self.args = args

        if args.controller_type == 'random_shooting':
            self.agent = RandomShooting(device,
                                        args.action_dim,
                                        self.observation_encoder,
                                        self.reward_model,
                                        self.rssm,
                                        args.horizon,
                                        args.num_control_samples,
                                        args.exploration_noise_var)
        elif args.controller_type == 'cem':
            self.agent = CEM(device,
                             args.action_dim,
                             self.observation_encoder,
                             self.reward_model,
                             self.rssm,
                             args.horizon,
                             args.num_control_samples,
                             args.max_iterations,
                             args.num_elites,
                             args.exploration_noise_var)
        else:
            raise ValueError('unknown controller type')

        self.sampler = Sampler(env, replay_buffer, self.agent)

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
                self.optimize_model()

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
        }, os.path.join(model_path, model_name))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.observation_encoder.load_state_dict(checkpoint['observation_encoder_state_dict'])
        self.observation_decoder.load_state_dict(checkpoint['observation_decoder_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.rssm.load_state_dict(checkpoint['rssm_state_dict'])
