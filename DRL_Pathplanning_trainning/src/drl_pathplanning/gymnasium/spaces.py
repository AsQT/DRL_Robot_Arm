"""
Action and observation space helpers for DRL_Pathplanning_trainning.
"""

import gymnasium.spaces as gym_spaces
import numpy as np


# --------------------------------------------------------------------------- #
# Constants — these match the environment's fixed design choices
# --------------------------------------------------------------------------- #
ACTION_DIM = 3  # (dx, dy, dz)
OBS_DIM = 15  # current(3) + target(3) + err(3) + rel_obs(3) + obs_size(3)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
OBS_DTYPE = np.float32


def make_action_space() -> gym_spaces.Box:
    """
    Returns the canonical action space for the Cartesian path planning task.

    Box([-1, -1, -1], [1, 1, 1], shape=(3,), dtype=np.float32)

    The 3-D normalised vector is scaled by ``action_step`` inside the
    environment to produce a Cartesian displacement.
    """
    return gym_spaces.Box(
        low=ACTION_LOW,
        high=ACTION_HIGH,
        shape=(ACTION_DIM,),
        dtype=OBS_DTYPE,
    )


def make_observation_space() -> gym_spaces.Box:
    """
    Returns the canonical observation space for the Cartesian path planning task.

    Box(-inf, inf, shape=(15,), dtype=np.float32)

    See CartesianPathPlanningEnv for the exact layout.
    """
    return gym_spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(OBS_DIM,),
        dtype=OBS_DTYPE,
    )
