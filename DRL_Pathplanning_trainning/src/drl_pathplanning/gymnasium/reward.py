"""
Simple distance-based reward for Cartesian path planning.

r_t = r_success + r_collision + r_distance + r_workspace + r_episode + r_shake

Designed to be general for random targets, random obstacles, and random obstacle sizes.
No geometric waypoints (P1/P2/P3), no phase logic, no safe-zone strategies.
"""

from __future__ import annotations

import numpy as np

from drl_pathplanning.gymnasium.config import RewardConfig


REWARD_COMPONENT_KEYS = (
    "success",
    "collision",
    "distance",
    "workspace",
    "episode",
    "time",
    "shake",
    "total",
)


class RewardCalculator:
    """
    Simple reward computation.

    Tracks action/movement history for shake penalty.
    Fully stateless for reward computation — all episode state lives in the env.
    """

    def __init__(self, reward_cfg: RewardConfig) -> None:
        self.cfg = reward_cfg
        self._action_history: list[np.ndarray] = []
        self._movement_history: list[np.ndarray] = []

    def reset(self) -> None:
        """Clear history at the start of each episode."""
        self._action_history.clear()
        self._movement_history.clear()

    def compute(
        self,
        prev_pos: np.ndarray,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        action: np.ndarray,
        is_success: bool,
        is_collision: bool,
        is_out_of_workspace: bool,
        is_timeout: bool,
        step_count: int,
        max_steps: int,
    ) -> tuple[float, dict]:
        """
        Compute reward for the current step.

        Returns
        -------
        tuple[float, dict]
            (total_reward, reward_components)
        """
        cfg = self.cfg
        prev_pos = np.asarray(prev_pos, dtype=np.float32)
        current_pos = np.asarray(current_pos, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        # Current distance to target
        d_t = float(np.linalg.norm(target_pos - current_pos))

        # ---- Terminal rewards ----
        valid_success = is_success and not is_collision and not is_out_of_workspace
        r_success = cfg.success_bonus if valid_success else 0.0
        r_collision = -cfg.collision_penalty if is_collision else 0.0
        r_workspace = -cfg.workspace_penalty if is_out_of_workspace else 0.0

        # ---- Episode/timeout penalty ----
        r_episode = -cfg.timeout_penalty if is_timeout else 0.0

        # ---- Distance reward ----
        r_distance = -cfg.distance_scale * d_t

        # ---- Time penalty per step ----
        r_time = -cfg.time_penalty if cfg.time_penalty > 0 else 0.0

        # ---- Shake penalty ----
        r_shake = 0.0
        if self._action_history:
            prev_action = self._action_history[-1]
            dot_val = float(np.dot(action, prev_action))
            norm_prod = float(np.linalg.norm(action) * np.linalg.norm(prev_action))
            if norm_prod > cfg.shake_min_movement:
                normed_dot = dot_val / norm_prod
                if normed_dot < cfg.shake_dot_threshold:
                    r_shake = -cfg.shake_penalty_scale

        # Accumulate history
        self._action_history.append(action.copy())
        movement = current_pos - prev_pos
        self._movement_history.append(movement.copy())

        # Trim history to shake_window
        if len(self._action_history) > cfg.shake_window:
            self._action_history.pop(0)
        if len(self._movement_history) > cfg.shake_window:
            self._movement_history.pop(0)

        # ---- Total ----
        total = (
            r_success
            + r_collision
            + r_distance
            + r_workspace
            + r_episode
            + r_time
            + r_shake
        )

        if not np.isfinite(total):
            total = -1e6

        components = {
            "success": float(r_success),
            "collision": float(r_collision),
            "distance": float(r_distance),
            "workspace": float(r_workspace),
            "episode": float(r_episode),
            "time": float(r_time),
            "shake": float(r_shake),
            "total": float(total),
        }

        return float(total), components
