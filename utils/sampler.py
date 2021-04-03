class Sampler:
    def __init__(self, env, replay_buffer, agent):
        super().__init__()

        self.env = env
        self.replay_buffer = replay_buffer
        self.agent = agent

    def collect_episodes(self, collect_episodes, exploration=False, random=False, render=False):
        self.agent.reset()
        all_episode_actions = []
        all_episode_observations = []
        all_episode_steps = []
        all_episode_rewards = []
        for _ in range(collect_episodes):
            episode_actions = []
            episode_observations = []
            episode_step = 0
            episode_reward = 0
            observation = self.env.reset()
            done = False
            while not done:
                if random:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.get_action(observation, exploration=exploration)
                episode_actions.append(action)
                episode_observations.append(observation)
                next_observation, reward, done, info = self.env.step(action)
                self.replay_buffer.push(observation, action, reward, done)
                observation = next_observation
                episode_step += 1
                episode_reward += reward
                if render:
                    self.env.render(height=400, width=400, camera_id=0)
            all_episode_actions.append(episode_actions)
            all_episode_observations.append(episode_observations)
            all_episode_steps.append(episode_step)
            all_episode_rewards.append(episode_reward)
        return all_episode_actions, all_episode_observations, all_episode_steps, all_episode_rewards
