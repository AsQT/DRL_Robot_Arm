"""
GP7 PyBullet Gymnasium Environment
==================================
Implements the active Gymnasium environment for training DRL agents on the
Yaskawa GP7 robotic manipulator using PyBullet for physics simulation.

This module exposes the :class:`GP7ReachPyBulletEnv` class, which is registered
as ``YaskawaGP7ReachPyBullet-{mode}-v0`` in :mod:`Industrial_Robotics_Gym`.
"""

import numpy as np
import typing as tp
import os
import gymnasium as gym
import pybullet as pb

from RoLE.Parameters.Robot import YASKAWA_GP7_Str
from RoLE.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
import PyBullet.Core
from config_loader import PROJECT_FOLDER_NAME

# --- Debug toggle: set to True to see per-reset and step-level debug output ---
ENABLE_DEBUG = False

# --- Warning toggle: set to True to see workspace/joint-limit step-level warnings ---
ENABLE_WARN = False


def _debug(msg: str) -> None:
    if ENABLE_DEBUG:
        print(msg)


def _warn(msg: str) -> None:
    if ENABLE_WARN:
        print(msg)

CONST_PROJECT_FOLDER = os.getcwd().split(PROJECT_FOLDER_NAME)[0] + PROJECT_FOLDER_NAME


class GP7ReachPyBulletEnv(gym.Env):
    """
    A Gymnasium reach environment for the Yaskawa GP7 robot, backed by PyBullet.

    The agent controls the robot by issuing normalised 3-D Cartesian delta actions
    (dx, dy, dz).  The environment computes the corresponding inverse kinematics
    (IK) solution via PyBullet's built-in IK solver and applies the joint commands
    to the simulated robot.

    Key design decisions:
        - TCP pose is read directly from PyBullet using the "ee_link" frame (resolved
          by name at startup; the old hardcoded link index 5 is no longer used).
          The internal RoLE FK model has a known ~0.35 m discrepancy and is bypassed.
        - Target orientation is fixed: quaternion [w=0, x=1, y=0, z=0] (180 deg
          around X), so the TCP Z-axis points world -Z (downward).
        - Real collision detection uses PyBullet's contact API against the
          Cube100 collision object.
        - In Collision-Free mode the obstacle position and size are included in
          the observation so the policy can learn to avoid it.

    Attributes:
        action_space: Box(-1, 1, shape=(3,)) — normalised Cartesian TCP delta.
        observation_space: Box(-inf, inf, shape=(15,)) — see observation layout below.

    Args:
        enable_gui: Show the PyBullet GUI window.  Set False for headless training.
        action_step: Scaling factor — each normalised action step moves the TCP by
            ``action_step`` metres.  Default 0.01 m.
        distance_thresh: Position error threshold for ``terminated == True``.
            Default 0.01 m (1 cm).
        max_episode_steps: Maximum :meth:`step` calls before ``truncated == True``.
            Default 200.
        env_mode: ``'Default'`` (no obstacle) or ``'Collision-Free'`` (cube present).

    Observation layout (15 dimensions):
        [tcp_x/y/z, target_x/y/z, err_x/y/z, rel_obs_x/y/z, obs_size_x/y/z]

        - tcp_x/y/z:    current TCP position in world frame (m)
        - target_x/y/z: current target position in world frame (m)
        - err_x/y/z:    target - tcp (m)
        - rel_obs_x/y/z: obstacle position relative to TCP (zero in Default mode)
        - obs_size_x/y/z: obstacle half-extents (zero in Default mode)

    Reward:
        Default:
            reward = -euclidean_distance(tcp_pos, target_pos)

        Collision-Free:
            reward = -(euclidean_distance(tcp_pos, target_pos)
                       + collision_obj_penalty * collision_obj_penalty_threshold)

            where:
                collision_obj_penalty = 1 / (1 + euclidean_distance(tcp_pos, obstacle_pos))

        Hard failures (no dense reward added):
            collision = -5.0  (real contact with obstacle, also truncates)
            workspace / IK / joint-limit failure = -1.0

        No success bonus is used.

    Termination / Truncation:
        - ``terminated == True``: TCP within ``distance_thresh`` of target.
        - ``truncated == True``: workspace violation, IK failure, joint limit breach,
          collision detected, or ``step_count >= max_episode_steps``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        enable_gui: bool = True,
        action_step: float = 0.01,
        distance_thresh: float = 0.03,
        max_episode_steps: int = 200,
        env_mode: str = 'Default',
    ) -> None:
        super().__init__()

        assert env_mode in ['Default', 'Collision-Free'], \
            f"env_mode must be 'Default' or 'Collision-Free', got '{env_mode}'"

        self.__enable_gui = enable_gui
        self.__action_step = float(action_step)
        self.__distance_thresh = float(distance_thresh)
        self.__max_episode_steps = max_episode_steps
        self.__episode_step = 0
        self.__env_mode = env_mode

        # Env_ID: 0 = no obstacle, 1 = with obstacle
        env_id = 0 if env_mode == 'Default' else 1
        print(f"[DEBUG] env_mode = {env_mode}")
        print(f"[DEBUG] env_id = {env_id}  (0=Default, 1=Collision-Free)")
        print(f"[DEBUG] obstacle enabled = {env_mode == 'Collision-Free'}")

        pybullet_env_props = {
            'Enable_GUI': enable_gui,
            'fps': 1000,
            'External_Base': None,
            'Env_ID': env_id,
            'Camera': {
                'Yaw': 70.0, 'Pitch': -32.0, 'Distance': 1.3,
                'Position': [0.05, -0.10, 0.06],
            },
        }

        self.__robot = PyBullet.Core.Robot_Cls(
            YASKAWA_GP7_Str,
            f'{CONST_PROJECT_FOLDER}/URDFs/Robots/{YASKAWA_GP7_Str.Name}/{YASKAWA_GP7_Str.Name}.urdf',
            pybullet_env_props,
        )

        self.__robot_id = self.__robot._Robot_Cls__robot_id

        # Resolve TCP link index by name ("ee_link") at runtime instead of hardcoding.
        # Core.py already resolves and prints this at startup; we look it up from the
        # same robot body.  The resolved index (likely 7) replaces the previous
        # hardcoded value of 5 (link_6).
        self.__tcp_link_index = self.__robot._Robot_Cls__Find_Link_Index_By_Name(
            self.__robot_id, "ee_link"
        )
        if self.__tcp_link_index is None:
            raise ValueError(
                "[ENV] ee_link not found in PyBullet link table. "
                "Check that the URDF defines ee_link and that URDF_MERGE_FIXED_LINKS "
                "is NOT set in Core.py robot loading."
            )
        # Backward-compatibility alias so existing __read_tcp_pose() needs no change.
        self.__link_index = self.__tcp_link_index
        print(f"[ENV] TCP link resolved: ee_link -> index {self.__tcp_link_index}")

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # --- Collision object setup: only load Cube100 in Collision-Free mode ---
        self.__collision_obj_T = None
        self.__collision_obj_size = None
        self.__collision_obj_id = None
        if self.__env_mode == 'Collision-Free':
            env_structure = PyBullet.Utilities.Get_Environment_Structure('YASKAWA_GP7', env_id)
            if env_structure.Collision_Object is not None:
                self.__collision_obj_T = env_structure.Collision_Object.T
                self.__collision_obj_scale = env_structure.Collision_Object.Scale
                self.__collision_obj_penalty_threshold = 0.01
                # Get collision object ID from robot's external objects
                self.__collision_obj_id = self.__robot._Robot_Cls__external_object.get('cube100_Collision', None)
                # Calculate obstacle size from AABB
                if self.__collision_obj_id is not None:
                    (min_aabb, max_aabb) = pb.getAABB(self.__collision_obj_id)
                    self.__collision_obj_size = np.array([
                        max_aabb[0] - min_aabb[0],
                        max_aabb[1] - min_aabb[1],
                        max_aabb[2] - min_aabb[2]
                    ], dtype=np.float32)
                    print(f'[ENV] Collision object loaded: id={self.__collision_obj_id}, size={self.__collision_obj_size}')
                else:
                    print('[ENV] Warning: Cube100 not found in robot external objects')
            else:
                print('[ENV] Warning: Collision-Free mode but no collision object defined in environment structure.')
        else:
            # Default mode: ensure no collision object is used
            self.__collision_obj_id = None
            self.__collision_obj_T = None
            self.__collision_obj_size = np.zeros(3, dtype=np.float32)
            print('[INFO] Default mode: Cube100 obstacle disabled')

        # Position-only IK for this Cartesian position-reaching environment.  Orientation is
        # not part of the observation/reward/success criteria.  6-D IK remains available
        # through use_orientation=True for debugging or future orientation-control tasks.
        self.__ik_props = {
            'delta_time': 0.01,
            'num_of_iteration': 500,
            'tolerance': 1e-30,
            'use_orientation': True,
            'ik_position_tolerance': 0.01,
        }

        # --- Get Search C-space (larger space, contains Home position) for workspace bounds ---
        search_C_vertices = self.__robot.Get_Configuration_Space_Vertices('Search')
        self.__ws_min = search_C_vertices.min(axis=0).astype(np.float32)
        self.__ws_max = search_C_vertices.max(axis=0).astype(np.float32)

        # Get Target C-space for target sampling
        target_C_vertices = self.__robot.Get_Configuration_Space_Vertices('Target')
        self.__target_min = target_C_vertices.min(axis=0).astype(np.float32)
        self.__target_max = target_C_vertices.max(axis=0).astype(np.float32)

        self.__target_pos = np.zeros(3, dtype=np.float32)
        self.__tcp_pos = np.zeros(3, dtype=np.float32)

        # --- Visualization: create target frame once (matches Core.py style) ---
        # The target frame 'T_EE_Target' is created here once and reused across episodes.
        # It is positioned in reset() via Transformation_External_Object.
        self.__robot.Add_External_Object(
            f'{CONST_PROJECT_FOLDER}/URDFs/Viewpoint/Viewpoint.urdf',
            'T_EE_Target',
            self.__build_target_htm(np.zeros(3)),  # initial pose at origin
            None,   # base parent link: None = world
            0.3,    # scale
            False   # not static (allows pose updates)
        )

        # --- Table is loaded automatically through Robot_Cls.__init__() via Scene_Object_Str ---

    # ------------------------------------------------------------------ #
    #   Fixed downward orientation for all targets.
    #   q = [w=0, x=1, y=0, z=0] => 180 deg around X => body Z → world -Z.
    # ------------------------------------------------------------------ #
    __Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # [w, x, y, z]

    # ------------------------------------------------------------------ #
    #   Internal helpers
    # ------------------------------------------------------------------ #
    def __read_tcp_pose(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        """Read TCP position and quaternion [w,x,y,z] directly from PyBullet."""
        ls = pb.getLinkState(self.__robot_id, self.__link_index, computeForwardKinematics=True)
        pos = np.array(ls[4], dtype=np.float32)
        quat_xyzw = np.array(ls[5], dtype=np.float32)  # [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)  # [w, x, y, z]
        return pos, quat_wxyz

    def __build_target_htm(self, position: np.ndarray) -> HTM_Cls:
        """Build 4x4 HTM with fixed Z-downward orientation at the given position."""
        return HTM_Cls(None, np.float64).Rotation(self.__class__.__Q_DOWN.tolist(), 'QUATERNION').Translation(position.tolist())

    def __sample_target_in_config_space(self) -> np.ndarray:
        """
        Sample a random target position within the GP7 Target configuration space.
        If in Collision-Free mode, also ensures target does not spawn inside the obstacle.
        """
        max_attempts = 100
        for _ in range(max_attempts):
            # Sample from Target C-space (not the larger Robot C-space)
            candidate = self.np_random.uniform(self.__target_min, self.__target_max).astype(np.float32)

            # Check if target is inside obstacle (Collision-Free mode)
            if self.__collision_obj_T is not None and self.__collision_obj_id is not None:
                obj_pos = self.__collision_obj_T.p.all().astype(np.float32)
                obj_half_size = self.__collision_obj_size / 2.0
                # Check if candidate is inside obstacle AABB (with small margin)
                margin = 0.02  # 2cm margin
                if (np.all(candidate >= obj_pos - obj_half_size - margin) and
                    np.all(candidate <= obj_pos + obj_half_size + margin)):
                    continue  # Target is inside obstacle, try again

            return candidate

        # Fallback: return center of Target C-space if couldn't find valid position
        center = ((self.__target_min + self.__target_max) / 2.0).astype(np.float32)
        print('[ENV] Warning: Could not find target outside obstacle, using Target C-space center')
        return center

    def __is_inside_workspace(self, pos: np.ndarray) -> bool:
        """Check if position is inside workspace bounds."""
        return np.all(pos >= self.__ws_min) and np.all(pos <= self.__ws_max)

    def __euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def __compute_reward(self, tcp_pos: np.ndarray, target_pos: np.ndarray) -> tp.Tuple[float, tp.Dict]:
        """
        Compute the dense reward for a normal (non-failure) step.

        Default mode:
            reward = -euclidean_distance(tcp_pos, target_pos)

        Collision-Free mode:
            reward = -(euclidean_distance(tcp_pos, target_pos)
                       + collision_obj_penalty * collision_obj_penalty_threshold)

        Args:
            tcp_pos:    Current TCP position (3,).
            target_pos: Current target position (3,).

        Returns:
            (reward, reward_info) where reward_info contains per-component floats.
        """
        distance = self.__euclidean(tcp_pos, target_pos)

        reward_distance = -distance
        reward_collision_soft = 0.0
        collision_dist = None
        collision_obj_penalty = 0.0

        if self.__env_mode == "Collision-Free" and self.__collision_obj_T is not None:
            obj_pos = self.__collision_obj_T.p.all().astype(np.float32)
            collision_dist = self.__euclidean(tcp_pos, obj_pos)
            collision_obj_penalty = 1.0 / (1.0 + collision_dist)
            reward_collision_soft = -collision_obj_penalty * self.__collision_obj_penalty_threshold

        reward = reward_distance + reward_collision_soft

        reward_info = {
            "distance": float(distance),
            "reward_distance": float(reward_distance),
            "reward_collision_soft": float(reward_collision_soft),
            "collision_soft_distance": None if collision_dist is None else float(collision_dist),
            "collision_soft_penalty": float(collision_obj_penalty),
        }

        return float(reward), reward_info

    def __check_collision(self) -> tp.Tuple[bool, int]:
        """
        Check for real collision between robot and collision object using PyBullet API.

        Returns:
            (1) is_collision [bool]: True if collision detected, False otherwise.
            (2) contact_count [int]: Number of contact points detected.
        """
        if self.__env_mode != 'Collision-Free' or self.__collision_obj_id is None:
            return False, 0

        # Use PyBullet contact API to detect collision
        contact_points = pb.getContactPoints(
            bodyA=self.__robot_id,
            bodyB=self.__collision_obj_id
        )
        contact_count = len(contact_points)
        is_collision = contact_count > 0

        return is_collision, contact_count

    def __normalize_obstacle_info(self, obj_pos: np.ndarray, obj_size: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Normalize obstacle position and size relative to TCP position.
        This helps the neural network learn better.

        Args:
            obj_pos: Obstacle position in world coordinates.
            obj_size: Obstacle size (half-extents).

        Returns:
            (1) normalized_relative_pos: Relative position from TCP to obstacle.
            (2) normalized_size: Normalized obstacle size.
        """
        relative_pos = obj_pos - self.__tcp_pos

        # Normalize relative position by workspace bounds (approximate)
        ws_range = self.__ws_max - self.__ws_min
        normalized_relative_pos = relative_pos / (ws_range + 1e-6)

        # Normalize size by workspace bounds
        normalized_size = obj_size / (ws_range + 1e-6)

        return normalized_relative_pos.astype(np.float32), normalized_size.astype(np.float32)

    # ------------------------------------------------------------------ #
    #   Gymnasium API
    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray) -> tp.Tuple[gym.spaces.Box, float, bool, bool, tp.Dict]:
        """
        Execute one environment step.

        The normalised action ``a ∈ [-1, 1]`` is scaled by ``action_step`` to produce a
        Cartesian delta (dx, dy, dz).  The environment then:
            1. Checks that the desired TCP position is within workspace bounds.
            2. Solves IK to find joint angles for that position.
            3. Commands the joints via PyBullet and steps the simulation.
            4. Reads the actual TCP pose from PyBullet (not the commanded pose).
            5. Checks for collision with the obstacle (Collision-Free mode).
            6. Computes reward and termination flags.

        Args:
            action: Normalised 3-D Cartesian delta ``[dx, dy, dz]`` ∈ [-1, 1]^3.

        Returns:
            observation: 15-D numpy array (see class docstring).
            reward: Float reward for this step.
            terminated: True if the task is successfully completed.
            truncated: True if the episode ended due to a failure condition.
            info: Dict with keys ``is_success``, ``distance``, ``is_collision``,
                ``contacts``, and ``termination_reason``.
        """

        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta = action * self.__action_step
        desired_pos = self.__tcp_pos + delta

        # --- Episode condition flags ---
        # Initialize: assume normal execution; IK failure or joint limit or workspace sets truncated.
        truncated = False
        reward = 0.0

        # --- Workspace check: terminate if robot goes outside bounds ---
        if not self.__is_inside_workspace(desired_pos):
            _warn('[ENV] Workspace violation: robot going outside bounds, episode truncated')
            truncated = True

        # --- IK solve ---
        if not truncated:
            T_target = self.__build_target_htm(desired_pos)
            (successful, theta) = self.__robot.Get_Inverse_Kinematics_Solution(
                T_target, self.__ik_props, enable_ghost=False
            )
            if not successful:
                print('[ENV] IK failed, episode truncated')
                truncated = True
            else:
                # --- Execute joint command ---
                theta_np = np.array(theta, dtype=np.float64)
                theta_limit = YASKAWA_GP7_Str.Theta.Limit
                for th_i, th_i_limit, th_index in zip(theta_np, theta_limit, self.__robot._Robot_Cls__theta_index):
                    if th_i_limit[0] <= th_i <= th_i_limit[1]:
                        pb.setJointMotorControl2(
                            self.__robot_id, th_index, pb.POSITION_CONTROL,
                            targetPosition=th_i, positionGain=1.0, velocityGain=1.0, force=100.0
                        )
                    else:
                        _warn('[ENV] Joint limit violation, episode truncated')
                        truncated = True
                        break
                if not truncated:
                    pb.stepSimulation()

        # --- Read actual TCP pose from PyBullet (always read) ---
        self.__tcp_pos, _ = self.__read_tcp_pose()

        # --- Check for real collision ---
        is_collision, contact_count = self.__check_collision()
        termination_reason = 'none'

        # --- Compute distance and termination ---
        pos_error = self.__target_pos - self.__tcp_pos
        dist = self.__euclidean(self.__tcp_pos, self.__target_pos)
        terminated = bool(dist < self.__distance_thresh)
        if terminated:
            termination_reason = 'success'

        # --- Compute reward based on failure type ---
        reward = 0.0
        reward_info = {
            "distance": float(dist),
            "reward_distance": 0.0,
            "reward_collision_soft": 0.0,
            "collision_soft_distance": None,
            "collision_soft_penalty": 0.0,
        }

        if is_collision:
            print(f'[ENV] COLLISION DETECTED! contacts={contact_count}')
            truncated = True
            termination_reason = 'collision'
            reward = -5.0

        if not is_collision:
            if truncated:
                if termination_reason == 'none':
                    termination_reason = 'workspace_limit'
                if reward == 0.0:
                    reward = -1.0
            else:
                reward, reward_info = self.__compute_reward(self.__tcp_pos, self.__target_pos)

        # --- Increment step counter and check max episode length ---
        self.__episode_step += 1
        truncated = truncated or bool(self.__episode_step >= self.__max_episode_steps)

        # --- Build 15D observation (same format for both modes) ---
        if self.__env_mode == 'Collision-Free' and self.__collision_obj_T is not None:
            obj_pos = self.__collision_obj_T.p.all().astype(np.float32)
            normalized_rel_pos, normalized_size = self.__normalize_obstacle_info(obj_pos, self.__collision_obj_size)
        else:
            normalized_rel_pos = np.zeros(3, dtype=np.float32)
            normalized_size = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([
            self.__tcp_pos,
            self.__target_pos,
            pos_error,
            normalized_rel_pos,
            normalized_size,
        ], axis=0).astype(np.float32)

        info = {
            'is_success': bool(terminated),
            'distance': float(dist),
            'is_collision': bool(is_collision),
            'contacts': int(contact_count),
            'termination_reason': str(termination_reason),
            'reward_distance': float(reward_info["reward_distance"]),
            'reward_collision_soft': float(reward_info["reward_collision_soft"]),
            'collision_soft_distance': reward_info["collision_soft_distance"],
            'collision_soft_penalty': float(reward_info["collision_soft_penalty"]),
        }

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: tp.Optional[int] = None,
        options: tp.Optional[tp.Dict] = None,
    ) -> tp.Tuple[gym.spaces.Box, tp.Dict]:
        """
        Reset the environment to the start of a new episode.

        Moves the robot to the Home joint configuration, samples a new random target
        within the Target configuration space, and returns the initial observation.
        If the sampled target would fall inside the obstacle (Collision-Free mode),
        the sampling retries up to 100 times before falling back to the space centre.

        Args:
            seed: Random seed passed to Gymnasium's internal seeder.
            options: Additional options dict (unused in this environment).

        Returns:
            observation: Initial 15-D observation (see class docstring).
            info: Dict with ``is_success`` set to False.
        """
        self.__episode_step = 0

        self.__robot.Reset('Home')
        self.__tcp_pos, _ = self.__read_tcp_pose()

        # --- Print workspace bounds and start TCP position once per reset ---
        _debug('[ENV] Workspace min: ' + str(self.__ws_min))
        _debug('[ENV] Workspace max: ' + str(self.__ws_max))
        _debug('[ENV] Start TCP position: ' + str(self.__tcp_pos))

        self.__target_pos = self.__sample_target_in_config_space()

        # --- Update target frame pose (created once in __init__) ---
        T_target = self.__build_target_htm(self.__target_pos)
        self.__robot.Transformation_External_Object('T_EE_Target', T_target, False)

        pos_error = self.__target_pos - self.__tcp_pos

        # --- Build 15D observation (same format for both modes) ---
        if self.__env_mode == 'Collision-Free' and self.__collision_obj_T is not None:
            obj_pos = self.__collision_obj_T.p.all().astype(np.float32)
            normalized_rel_pos, normalized_size = self.__normalize_obstacle_info(obj_pos, self.__collision_obj_size)
        else:
            normalized_rel_pos = np.zeros(3, dtype=np.float32)
            normalized_size = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([
            self.__tcp_pos,
            self.__target_pos,
            pos_error,
            normalized_rel_pos,
            normalized_size,
        ], axis=0).astype(np.float32)

        _debug('[OBS] shape = ' + str(obs.shape))

        return obs, {
            'is_success': False,
            'distance': float(self.__euclidean(self.__tcp_pos, self.__target_pos)),
            'is_collision': False,
            'contacts': 0,
            'termination_reason': 'none',
            'reward_distance': 0.0,
            'reward_collision_soft': 0.0,
            'collision_soft_distance': None,
            'collision_soft_penalty': 0.0,
        }

    def close(self) -> None:
        """Disconnect from the PyBullet physics server and clean up resources."""
        self.__robot.Disconnect()
