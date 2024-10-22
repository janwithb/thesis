import numpy as np
from mujoco_py.modder import MaterialModder, TextureModder, LightModder, CameraModder

from envs.fetch import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, randomize_light, randomize_camera,
        randomize_target_color, randomize_target_size, randomize_table_color, randomize_floor_color,
        randomize_background_color, randomize_robot_color
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.randomize_light = randomize_light
        self.randomize_camera = randomize_camera
        self.randomize_target_color = randomize_target_color
        self.randomize_target_size = randomize_target_size
        self.randomize_table_color = randomize_table_color
        self.randomize_floor_color = randomize_floor_color
        self.randomize_background_color = randomize_background_color
        self.randomize_robot_color = randomize_robot_color

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

        self.material_modder = MaterialModder(self.sim)
        self.texture_modder = TextureModder(self.sim)
        self.light_modder = LightModder(self.sim)
        self.camera_modder = CameraModder(self.sim)

        self.base_distance = 1.785
        self.base_azimuth = 180.
        self.base_elevation = -25.

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:torso_fixed_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        self.viewer.cam.distance = self.base_distance
        self.viewer.cam.azimuth = self.base_azimuth
        self.viewer.cam.elevation = self.base_elevation

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        # set colors
        first_color = [0.356, 0.361, 0.376, 1.]
        table_color = [0.46, 0.87, 0.58, 1.]
        third_color_geoms = ['robot0:torso_lift_link', 'robot0:shoulder_pan_link', 'robot0:upperarm_roll_link',
                             'robot0:forearm_roll_link', 'robot0:wrist_roll_link', 'robot0:gripper_link']
        for name in third_color_geoms:
            geom_id = self.sim.model.geom_name2id(name)
            self.sim.model.geom_rgba[geom_id] = first_color
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('table_mat0')] = table_color

        # set sizes
        size = [0.05, 0.05, 0.05]
        self.sim.model.site_size[self.sim.model.site_name2id('target0')] = size

        # randomize colors
        self._set_random_colors()

        # randomize target size
        if self.randomize_target_size:
            self.sim.model.site_size[self.sim.model.site_name2id('target0')] = size + np.random.uniform(-0.02, 0.02, 3)

        # randomize light
        if self.randomize_light:
            self._set_random_lights()

        # randomize camera
        if self.randomize_camera:
            self._set_random_camera()

        self.sim.forward()
        return True

    def _set_random_colors(self):
        first_color = list(np.append(np.random.uniform(0, 1, 3), [1.]))
        second_color = list(np.append(np.random.uniform(0, 1, 3), [1.]))
        third_color = list(np.append(np.random.uniform(0, 1, 3), [1.]))
        target_color = list(np.append(np.random.uniform(0, 1, 3), [1.]))
        table_color = list(np.append(np.random.uniform(0, 1, 3), [1.]))
        floor_color = list(np.append(np.random.uniform(0, 1, 3), [1.]))

        first_color_geoms = ['robot0:base_link', 'robot0:head_pan_link', 'robot0:r_gripper_finger_link',
                             'robot0:l_gripper_finger_link']
        second_color_geoms = ['robot0:head_tilt_link', 'robot0:shoulder_lift_link', 'robot0:elbow_flex_link',
                              'robot0:wrist_flex_link', 'robot0:torso_fixed_link']
        third_color_geoms = ['robot0:torso_lift_link', 'robot0:shoulder_pan_link', 'robot0:upperarm_roll_link',
                             'robot0:forearm_roll_link', 'robot0:wrist_roll_link', 'robot0:gripper_link']

        if self.randomize_target_color:
            self.sim.model.site_rgba[self.sim.model.site_name2id('target0')] = target_color
        if self.randomize_table_color:
            self.sim.model.geom_rgba[self.sim.model.geom_name2id('table_mat0')] = table_color
            # randomize reflectance
            self.material_modder.set_reflectance('table_mat0', np.random.uniform(0.0, 0.5))
        if self.randomize_floor_color:
            self.sim.model.geom_rgba[self.sim.model.geom_name2id('floor0')] = floor_color
        if self.randomize_background_color:
            self.texture_modder.rand_noise('skybox')
        if self.randomize_robot_color:
            for name in first_color_geoms:
                geom_id = self.sim.model.geom_name2id(name)
                self.sim.model.geom_rgba[geom_id] = first_color
            for name in second_color_geoms:
                geom_id = self.sim.model.geom_name2id(name)
                self.sim.model.geom_rgba[geom_id] = second_color
            for name in third_color_geoms:
                geom_id = self.sim.model.geom_name2id(name)
                self.sim.model.geom_rgba[geom_id] = third_color

    def _set_random_lights(self):
        amb_base = np.random.uniform(0.1, 0.8)
        amb_delr = np.random.uniform(-0.05, 0.05)
        amb_delg = np.random.uniform(-0.05, 0.05)
        amb_delb = np.random.uniform(-0.05, 0.05)
        amb = [amb_base + amb_delr, amb_base + amb_delg, amb_base + amb_delb]

        dif_base = np.random.uniform(0.4, 0.9)
        dif_delr = np.random.uniform(-0.1, 0.1)
        dif_delg = np.random.uniform(-0.1, 0.1)
        dif_delb = np.random.uniform(-0.1, 0.1)
        dif = [dif_base + dif_delr, dif_base + dif_delg, dif_base + dif_delb]

        spec_base = np.random.uniform(0.05, 0.3)
        spec_delr = np.random.uniform(-0.02, 0.02)
        spec_delg = np.random.uniform(-0.02, 0.02)
        spec_delb = np.random.uniform(-0.02, 0.02)
        spec = [spec_base + spec_delr, spec_base + spec_delg, spec_base + spec_delb]

        self.light_modder.set_ambient('light0', amb)
        self.light_modder.set_diffuse('light0', dif)
        self.light_modder.set_specular('light0', spec)

    def _set_random_camera(self):
        distance = np.random.uniform(0., 0.05)
        azimuth = np.random.uniform(-2, 2)
        elevation = np.random.uniform(-2, 2)
        self.viewer.cam.distance = self.base_distance + distance
        self.viewer.cam.azimuth = self.base_azimuth + azimuth
        self.viewer.cam.elevation = self.base_elevation + elevation

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            if not self.target_in_the_air:
                goal[2] = self.initial_gripper_xpos[2] - 0.1
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500, **kwargs):
        return super(FetchEnv, self).render(mode, width, height)
