from gym.envs.registration import register


register(
    id='FetchReach-v2',
    entry_point='envs.fetch.reach:FetchReachEnv',
    kwargs={
        'reward_type': 'dense',
        'randomize_env': False,
    },
    max_episode_steps=250,
)

register(
    id='FetchReachRandom-v2',
    entry_point='envs.fetch.reach:FetchReachEnv',
    kwargs={
        'reward_type': 'dense',
        'randomize_env': True,
    },
    max_episode_steps=250,
)
