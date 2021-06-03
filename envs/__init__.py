from gym.envs.registration import register


register(
    id='FetchReach-v2',
    entry_point='envs.fetch.reach:FetchReachEnv',
    kwargs={
        'reward_type': 'dense',
        'randomize_light': False,
        'randomize_camera': False,
        'randomize_target_color': False,
        'randomize_target_size': False,
        'randomize_table_color': False,
        'randomize_floor_color': False,
        'randomize_background_color': False,
        'randomize_robot_color': False
    },
    max_episode_steps=250,
)

register(
    id='FetchReachRandomLight-v2',
    entry_point='envs.fetch.reach:FetchReachEnv',
    kwargs={
        'reward_type': 'dense',
        'randomize_light': True,
        'randomize_camera': False,
        'randomize_target_color': False,
        'randomize_target_size': False,
        'randomize_table_color': False,
        'randomize_floor_color': False,
        'randomize_background_color': False,
        'randomize_robot_color': False
    },
    max_episode_steps=250,
)

register(
    id='FetchReachRandomCamera-v2',
    entry_point='envs.fetch.reach:FetchReachEnv',
    kwargs={
        'reward_type': 'dense',
        'randomize_light': False,
        'randomize_camera': True,
        'randomize_target_color': False,
        'randomize_target_size': False,
        'randomize_table_color': False,
        'randomize_floor_color': False,
        'randomize_background_color': False,
        'randomize_robot_color': False
    },
    max_episode_steps=250,
)

register(
    id='FetchReachRandomTarget-v2',
    entry_point='envs.fetch.reach:FetchReachEnv',
    kwargs={
        'reward_type': 'dense',
        'randomize_light': False,
        'randomize_camera': False,
        'randomize_target_color': True,
        'randomize_target_size': True,
        'randomize_table_color': False,
        'randomize_floor_color': False,
        'randomize_background_color': False,
        'randomize_robot_color': False
    },
    max_episode_steps=250,
)

register(
    id='FetchReachRandomFloor-v2',
    entry_point='envs.fetch.reach:FetchReachEnv',
    kwargs={
        'reward_type': 'dense',
        'randomize_light': False,
        'randomize_camera': False,
        'randomize_target_color': False,
        'randomize_target_size': False,
        'randomize_table_color': True,
        'randomize_floor_color': True,
        'randomize_background_color': False,
        'randomize_robot_color': False
    },
    max_episode_steps=250,
)

register(
    id='FetchReachRandomBackground-v2',
    entry_point='envs.fetch.reach:FetchReachEnv',
    kwargs={
        'reward_type': 'dense',
        'randomize_light': False,
        'randomize_camera': False,
        'randomize_target_color': False,
        'randomize_target_size': False,
        'randomize_table_color': False,
        'randomize_floor_color': False,
        'randomize_background_color': True,
        'randomize_robot_color': False
    },
    max_episode_steps=250,
)

register(
    id='FetchReachRandomRobot-v2',
    entry_point='envs.fetch.reach:FetchReachEnv',
    kwargs={
        'reward_type': 'dense',
        'randomize_light': False,
        'randomize_camera': False,
        'randomize_target_color': False,
        'randomize_target_size': False,
        'randomize_table_color': False,
        'randomize_floor_color': False,
        'randomize_background_color': False,
        'randomize_robot_color': True
    },
    max_episode_steps=250,
)
