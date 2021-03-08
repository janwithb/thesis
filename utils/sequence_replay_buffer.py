import os
import pickle
import random

import numpy as np
from six.moves import xrange


class Episode(object):
    def __init__(self, observations, actions, rewards, starting_state):
        self.states = observations
        self.actions = actions
        self.rewards = rewards
        self.starting_state = starting_state


class SequenceReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = {}
        self.init_length = 0

    def __len__(self):
        return self.cur_size

    def add(self, episodes):
        idx = 0
        while self.cur_size < self.max_size and idx < len(episodes):
            self.buffer[self.cur_size] = episodes[idx]
            self.cur_size += 1
            idx += 1

        if idx < len(episodes):
            remove_idxs = self.remove_n(len(episodes) - idx)
            for remove_idx in remove_idxs:
                self.buffer[remove_idx] = episodes[idx]
                idx += 1

        assert len(self.buffer) == self.cur_size

    def remove_n(self, n):
        """Get n items for removal."""
        # random removal
        idxs = random.sample(xrange(self.init_length, self.cur_size), n)
        return idxs

    def get_batch(self, n):
        """Get batch of episodes to train on."""
        # random batch
        idxs = random.sample(xrange(self.cur_size), n)
        return np.array([self.buffer[idx] for idx in idxs])

    def get_chunk_batch(self, n, t):
        """Get batch of episodes with length t to train on."""
        # random batch
        idxs = random.sample(xrange(self.cur_size), n)
        batch = np.array([self.buffer[idx] for idx in idxs])
        for episode in batch:
            start_idx = random.randint(0, len(episode.states) - t)
            states_chunk = np.array(episode.states)[start_idx:start_idx + t]
            actions_chunk = np.array(episode.actions)[start_idx:start_idx + t]
            rewards_chunk = np.array(episode.rewards)[start_idx:start_idx + t]
            episode.states = states_chunk
            episode.actions = actions_chunk
            episode.rewards = rewards_chunk
            if start_idx != 0:
                episode.starting_state = episode.states[start_idx - 1]
        return batch

    def save(self, save_dir, save_name):
        payload = self.buffer
        with open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(payload, f)

    def load(self, load_dir):
        with open(os.path.join(load_dir), 'rb') as f:
            payload = pickle.load(f)
            episodes = np.array([payload[idx] for idx in range(len(payload))])
            self.add(episodes)
