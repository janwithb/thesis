import os
import time

import torch

from tqdm import tqdm
from algos.dreamer_base import DreamerBase
from policies.cem import CEM
from policies.random_shooting import RandomShooting


class DreamerMPC(DreamerBase):
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
                 action_repeat=1,
                 controller_type='random_shooting',
                 action_space=None,
                 horizon=20,
                 num_control_samples=200,
                 max_iterations=3,
                 num_elites=20):

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
                         action_repeat=action_repeat)

        if controller_type == 'random_shooting':
            self.controller = RandomShooting(
                device,
                action_space,
                self.rollout,
                self.reward_model,
                horizon,
                num_control_samples
            )
        elif controller_type == 'cem':
            self.controller = CEM(
                device,
                action_space,
                self.rollout,
                self.reward_model,
                horizon,
                num_control_samples,
                max_iterations,
                num_elites
            )
        else:
            raise ValueError('unknown controller type')

    def train(self,
              init_episodes=100,
              init_episode_length=100,
              policy_episodes=10,
              policy_episode_length=100,
              random_episodes=10,
              random_episode_length=100,
              training_iterations=100,
              model_iterations=100,
              batch_size=64,
              chunk_size=50,
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
            self.training = True
            self.eval = False

            # model training loop
            for _ in tqdm(range(model_iterations), desc='Model Training'):
                samples = self.replay_buffer.get_chunk_batch(batch_size, chunk_size)
                self.optimize_model(samples)
                self.model_itr += 1

            # collect new random data
            rand_episodes, rand_steps = self.sampler.collect_random_episodes(random_episodes, random_episode_length,
                                                                             render=render_training)

            # collect on policy data
            pol_episodes, pol_steps = self.sampler.collect_policy_episodes(policy_episodes, policy_episode_length,
                                                                           self.mpc_policy,
                                                                           self.get_state_representation,
                                                                           self.device,
                                                                           render=render_training)

            episodes = rand_episodes + pol_episodes
            total_steps = (rand_steps + pol_steps) * self.action_repeat

            self.replay_buffer.add(episodes)
            self.step += total_steps

            # save model frequently
            if save_iter_model and it % save_iter_model_freq == 0:
                self.save_model(model_dir, 'model_iter_' + str(it))

            itr_time = time.time() - itr_start_time
            self.logger.log('train/itr_time', itr_time, self.step)
            self.itr += 1

            # evaluate policy
            if it % eval_freq == 0:
                self.evaluate(eval_episodes, eval_episode_length, self.mpc_policy,
                              save_eval_video, video_dir, render_eval)

    def mpc_policy(self, state):
        action = self.controller.get_action(state)
        return action

    def save_model(self, model_path, model_name):
        torch.save({
            'observation_encoder_state_dict': self.observation_encoder.state_dict(),
            'observation_decoder_state_dict': self.observation_decoder.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'representation_state_dict': self.representation.state_dict(),
            'transition_state_dict': self.transition.state_dict()
        }, os.path.join(model_path, model_name))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.observation_encoder.load_state_dict(checkpoint['observation_encoder_state_dict'])
        self.observation_decoder.load_state_dict(checkpoint['observation_decoder_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.representation.load_state_dict(checkpoint['representation_state_dict'])
        self.transition.load_state_dict(checkpoint['transition_state_dict'])
