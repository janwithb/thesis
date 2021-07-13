import os
import pickle

import numpy as np


class SequenceReplayBuffer(object):
    """
    Replay buffer for data sequences.
    """
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        """
        Add experience to replay buffer
        NOTE: observation should be transformed to np.uint8 before push
        """
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        """
        Sample experiences from replay buffer (almost) uniformly
        The resulting array will be of the form (batch_size, chunk_length)
        and each batch is consecutive sequence
        NOTE: too large chunk_length for the length of episode will cause problems
        """
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:])
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1])
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        return sampled_observations, sampled_actions, sampled_rewards, sampled_done

    def save(self, save_dir, save_name):
        payload = {'observations': self.observations,
                   'actions': self.actions,
                   'rewards': self.rewards,
                   'done': self.done,
                   'index': self.index,
                   'is_filled': self.is_filled}
        with open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(payload, f)

    def load(self, load_dir):
        with open(os.path.join(load_dir), 'rb') as f:
            payload = pickle.load(f)
            self.observations = payload['observations']
            self.actions = payload['actions']
            self.rewards = payload['rewards']
            self.done = payload['done']
            self.index = payload['index']
            self.is_filled = payload['is_filled']

    def __len__(self):
        return self.capacity if self.is_filled else self.index
