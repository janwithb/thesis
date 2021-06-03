import os
from gym import utils
from envs.fetch import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', randomize_light=False, randomize_camera=False,
                 randomize_target_color=False, randomize_target_size=False, randomize_table_color=False,
                 randomize_floor_color=False, randomize_background_color=False, randomize_robot_color=False):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.25, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, randomize_light=randomize_light,
            randomize_camera=randomize_camera, randomize_target_color=randomize_target_color,
            randomize_target_size=randomize_target_size, randomize_table_color=randomize_table_color,
            randomize_floor_color=randomize_floor_color, randomize_background_color=randomize_background_color,
            randomize_robot_color=randomize_robot_color)
        utils.EzPickle.__init__(self)
