import os

import torch

from tqdm import tqdm
from utils.sequence_replay_buffer import Episode
from utils.video_recorder import VideoRecorder


class Sampler:
    def __init__(self, env, video_recorder=None):
        super().__init__()

        self.env = env
        self.video_recorder = video_recorder

    def collect_random_episodes(self, collect_episodes, collect_episode_length, render=False):
        episodes = []
        total_steps = 0
        for _ in tqdm(range(collect_episodes), desc='Collect episodes'):
            observation, done = self.env.reset(), False
            starting_state = observation
            observations, actions, rewards = [], [], []
            for t in range(collect_episode_length):
                action = self.env.action_space.sample()
                next_observation, reward, done, info = self.env.step(action)
                observations.append(next_observation)
                actions.append(action)
                rewards.append(reward)
                if render:
                    self.env.unwrapped.render()
                if self.video_recorder is not None:
                    self.video_recorder.capture_frame()
                total_steps += 1
            episode = Episode(observations, actions, rewards, starting_state)
            episodes.append(episode)
        return episodes, total_steps

    def collect_policy_episodes(self, collect_episodes, collect_episode_length, policy, representation,
                                device, render=False):
        episodes = []
        total_steps = 0
        for _ in tqdm(range(collect_episodes), desc='Collect episodes'):
            observation, done = self.env.reset(), False
            starting_state = observation
            observations, actions, rewards = [], [], []
            rssm_state = representation(torch.as_tensor(observation, device=device), None, None)
            for t in range(collect_episode_length):
                action = policy(rssm_state)
                next_observation, reward, done, info = self.env.step(action)
                rssm_state = representation(torch.as_tensor(next_observation, device=device),
                                            torch.as_tensor(action, device=device),
                                            rssm_state)
                observations.append(next_observation)
                actions.append(action)
                rewards.append(reward)
                if render:
                    self.env.unwrapped.render()
                if self.video_recorder is not None:
                    self.video_recorder.capture_frame()
                total_steps += 1
            episode = Episode(observations, actions, rewards, starting_state)
            episodes.append(episode)
        return episodes, total_steps

    def reset_video_recorder(self, video_dir, video_name):
        # close any existing video recorder
        if self.video_recorder:
            self.video_recorder.close()

        # start recording the next video
        self.video_recorder = VideoRecorder(
            self.env,
            base_path=os.path.join(video_dir, video_name),
            enabled=True
        )
