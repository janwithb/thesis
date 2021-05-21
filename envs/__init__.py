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
        'randomize_light': False,
        'randomize_camera': False,
        'randomize_target_color': False,
        'randomize_table_color': False,
        'randomize_floor_color': False,
        'randomize_background_color': True,
        'randomize_robot_color': False
    },
    max_episode_steps=250,
)
