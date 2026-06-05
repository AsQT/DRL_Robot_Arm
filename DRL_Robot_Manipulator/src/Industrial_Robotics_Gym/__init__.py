"""
GP7 PyBullet Gymnasium Environment Package
=========================================
Registers Gymnasium environments for the Yaskawa GP7 robotic manipulator.

All environments are built on top of PyBullet for physics simulation and expose
the standard Gymnasium API (reset / step / close).

Available environments:

    YaskawaGP7ReachPyBullet-Default-v0
        Free-space reaching task with no obstacle.

    YaskawaGP7ReachPyBullet-Collision-Free-v0
        Reaching task with a cube obstacle present.  The agent must navigate
        around the obstacle to reach the target.

Both environments share the same action space (3D Cartesian delta) and the
same 15-dimensional observation space.  They differ only in whether the
collision object is loaded.

Usage::

    import gymnasium as gym
    import Industrial_Robotics_Gym

    env = gym.make(
        "YaskawaGP7ReachPyBullet-Collision-Free-v0",
        enable_gui=True,           # True = PyBullet GUI window, False = headless
        action_step=0.01,          # metres per normalised action step
        distance_thresh=0.01,      # success threshold in metres
        max_episode_steps=200,
    )

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
"""

import gymnasium as gym
from Industrial_Robotics_Gym.Environment.GP7ReachPyBulletEnv import GP7ReachPyBulletEnv

# Supported environment modes.
_ENV_MODES = ['Default', 'Collision-Free']

# Register environments. Only 'env_mode' is fixed; all other constructor parameters
# (enable_gui, action_step, distance_thresh, max_episode_steps) are overridden at
# gym.make() call time.
for _mode in _ENV_MODES:
    _env_id = f'YaskawaGP7ReachPyBullet-{_mode}-v0'
    if _env_id not in gym.registry:
        gym.register(
            id=_env_id,
            entry_point='Industrial_Robotics_Gym.Environment.GP7ReachPyBulletEnv:GP7ReachPyBulletEnv',
            kwargs={'env_mode': _mode},
            max_episode_steps=200,
        )
