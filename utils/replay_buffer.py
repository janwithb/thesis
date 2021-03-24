import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_size, action_size, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = torch.empty((capacity, obs_size), device=device)
        self.next_obses = torch.empty((capacity, obs_size), device=device)
        self.actions = torch.empty((capacity, action_size), device=device)
        self.rewards = torch.empty((capacity, 1), device=device)
        self.not_dones = torch.empty((capacity, 1), device=device)
        self.not_dones_no_max = torch.empty((capacity, 1), device=device)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        self.obses[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obses[self.idx] = next_obs
        self.not_dones[self.idx] = not done
        self.not_dones_no_max[self.idx] = not done_no_max

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        not_dones = self.not_dones[idxs]
        not_dones_no_max = self.not_dones_no_max[idxs]

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
