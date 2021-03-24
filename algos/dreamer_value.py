import os
import time

import numpy as np
import torch

from tqdm import tqdm
from algos.dreamer_base import DreamerBase
from models.actor import ActorModel
from models.dense import DenseModel
from models.rssm import get_feat, RSSMState
from utils.misc import get_parameters, FreezeParameters, compute_return, flatten_rssm_state


class DreamerValue(DreamerBase):
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
                 actor_lr=8e-5,
                 value_lr=8e-5,
                 grad_clip=100.0,
                 free_nats=3,
                 kl_scale=1,
                 action_repeat=1,
                 value_shape=None,
                 value_layers=3,
                 value_hidden=200,
                 actor_layers=3,
                 actor_hidden=200,
                 actor_dist='tanh_normal',
                 imagine_horizon=15,
                 discount=0.99,
                 discount_lambda=0.95,
                 train_noise=0.3,
                 eval_noise=0.0,
                 expl_method='additive_gaussian',
                 expl_amount=0.3,
                 expl_decay=0.0,
                 expl_min=0.0):

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

        # actor model
        self.actor_model = ActorModel(self.action_size, self.feature_size, actor_hidden, actor_layers, actor_dist)

        # value model
        self.value_model = DenseModel(self.feature_size, value_shape, value_layers, value_hidden)

        # bundle models
        self.actor_modules = [self.actor_model]
        self.value_modules = [self.value_model]

        # gpu settings
        self.actor_model.to(self.device)
        self.value_model.to(self.device)

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

    def train(self,
              init_episodes=100,
              init_episode_length=100,
              iter_episodes=10,
              iter_episode_length=100,
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
            self.set_mode('train')

            # model training loop
            for _ in tqdm(range(model_iterations), desc='Model Training'):
                samples = self.replay_buffer.get_chunk_batch(batch_size, chunk_size)
                self.optimize(samples)
                self.model_itr += 1

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

    def optimize(self, samples):
        post = self.optimize_model(samples)
        imag_feat, discount, returns = self.optimize_actor(post)
        self.optimize_value(imag_feat, discount, returns)

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
        if self.model_itr % self.tensorboard_log_freq == 0:
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
        if self.model_itr % self.tensorboard_log_freq == 0:
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
