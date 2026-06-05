"""
drl_pathplanning — Frame-only Cartesian DRL path planning.

This package is FRAME-ONLY:
- NO robot model, NO URDF, NO IK/FK, NO joints, NO MoveIt, NO ROS2.
- PyBullet is used ONLY for optional visualisation (pybullet/).
- The Gymnasium environment (gymnasium/) is the source of truth.

Subpackages
-----------
gymnasium    — Gymnasium environment (no PyBullet dependency)
pybullet    — Optional PyBullet visualisation (lazy import)
geometry    — Distance, path metrics, workspace, collision geometry
training    — Trainer, env factory, callbacks, curriculum
"""

try:
    import gymnasium as gym
    _ID = "CartesianPathPlanning-Default-v0"
    if _ID not in gym.registry:
        gym.register(
            id=_ID,
            entry_point="drl_pathplanning.gymnasium.cartesian_frame_env:CartesianPathPlanningEnv",
        )
except Exception:
    pass

__all__ = []
