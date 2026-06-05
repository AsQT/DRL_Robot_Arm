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

# Stable-Baselines3 (A set of implementations of reinforcement learning algorithms in PyTorch) [pip3 install stable-baselines3]
import stable_baselines3.common.env_checker

# Import the package to register environments.
import Industrial_Robotics_Gym

# The name of the environment mode.
#   'Default':
#       The mode called "Default" demonstrates an environment without a collision object.
#   'Collision-Free':
#       The mode called "Collision-Free" demonstrates an environment with a collision object.
CONST_ENV_MODE = 'Default'


def main():
    # Create the environment via gym.make().
    gym_environment = gym.make(
        f"YaskawaGP7ReachPyBullet-{CONST_ENV_MODE}-v0",
        enable_gui=False,
        action_step=0.02,
        distance_thresh=0.01,
        max_episode_steps=200,
    )

    # Verify that the custom environment adheres to the gym API and is compatible with Stable Baselines3 (SB3).
    stable_baselines3.common.env_checker.check_env(gym_environment, warn=True)


if __name__ == '__main__':
    sys.exit(main())
