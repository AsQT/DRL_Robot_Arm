# System (Default)
import sys
import os

# Find the project root by searching upward for the 'src' directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = None
for _parent in (_SCRIPT_DIR, None):
    if _parent is None:
        break
    _check = os.path.join(_parent, 'src')
    if os.path.isdir(_check):
        _SRC_DIR = _check
        break
    _next = os.path.dirname(_parent)
    if _next == _parent:
        break
    _parent = _next
if _SRC_DIR is None:
    _SRC_DIR = os.path.join(os.getcwd(), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Gymnasium (Developing and comparing reinforcement learning algorithms) [pip3 install gymnasium]
import gymnasium as gym

# Import the package to register environments.
import Industrial_Robotics_Gym

# The name of the environment mode.
#   'Default':
#       The mode called "Default" demonstrates an environment without a collision object.
#   'Collision-Free':
#       The mode called "Collision-Free" demonstrates an environment with a collision object.
CONST_ENV_MODE = 'Default'

# Information about whether the target is selected statically or randomly.
CONST_STATIC_TARGET = False


def main():
    # Create the environment via gym.make().
    gym_environment = gym.make(
        f"YaskawaGP7ReachPyBullet-{CONST_ENV_MODE}-v0",
        enable_gui=True,
        action_step=0.02,
        distance_thresh=0.01,
        max_episode_steps=200,
    )

    # Reset the pre-defined environment of the gym.
    observations, informations = gym_environment.reset()

    for _ in range(1000):
        # Obtain a random action sample from the entire action space.
        action = gym_environment.action_space.sample()

        # Perform the action within the pre-defined environment and get the new observation space.
        observations, reward, terminated, truncated, informations = gym_environment.step(action)

        # When the reach task process is terminated or truncated, reset the pre-defined gym environment.
        if terminated or truncated:
            observations, informations = gym_environment.reset()

    # Disconnect the created environment from a physical server.
    gym_environment.close()


if __name__ == '__main__':
    sys.exit(main())
