"""
Callback utilities for Stable-Baselines3 training.

Provides episode-level logging callbacks that track rewards, lengths,
distances, success rates, termination reasons, action diagnostics,
and per-component reward terms during training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import stable_baselines3.common.callbacks as sb3_cb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drl_pathplanning.training.curriculum import CurriculumTargetSampler
    from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv

from drl_pathplanning.gymnasium.reward import REWARD_COMPONENT_KEYS


def _safe_mean(values, default=float("nan")):
    """Safe mean that handles empty/None input gracefully."""
    if values is None:
        return float(default)
    vals = list(values)
    return float(np.mean(vals)) if vals else float(default)


class EpisodeCallback(sb3_cb.BaseCallback):
    """
    Episode-level progress logger for SB3 training.

    Tracks cumulative reward, episode length, final distance, success, and
    termination reasons for each episode.  Prints a formatted progress line
    every ``episode_log_interval`` episodes showing rolling averages.
    """

    def __init__(
        self,
        log_interval: int = 10,
        episode_log_interval: int = 1000,
        success_threshold: float = 0.01,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_interval = log_interval
        self.episode_log_interval = episode_log_interval
        self.success_threshold = success_threshold

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_distances: list[float] = []
        self.episode_successes: list[bool] = []
        self.episode_initial_distances: list[float] = []
        self.episode_termination_reasons: list[str] = []
        self.episode_expected_path_lengths: list[float] = []
        self.episode_actual_path_lengths: list[float] = []

        for _key in REWARD_COMPONENT_KEYS:
            setattr(self, f"episode_{_key}", [])

        self._current_rewards: list[float] = []
        self._current_distances: list[float] = []
        self._current_success_flags: list[bool] = []
        self._current_action_norms: list[float] = []
        for _key in REWARD_COMPONENT_KEYS:
            setattr(self, f"_current_{_key}", [])
        self._initial_distance: float = 0.0

    def _on_training_start(self) -> None:
        self._initial_distance = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if rewards is not None and len(rewards) > 0:
            self._current_rewards.append(float(rewards[0]))

        if infos is not None and len(infos) > 0:
            info = infos[0]
            if "distance" in info:
                self._current_distances.append(float(info["distance"]))
            if "is_success" in info:
                self._current_success_flags.append(bool(info["is_success"]))
            if "action_norm" in info:
                self._current_action_norms.append(float(info["action_norm"]))

            rc = info.get("reward_components", {})
            for key in REWARD_COMPONENT_KEYS:
                val = rc.get(key, 0.0)
                getattr(self, f"_current_{key}").append(float(val))

        if dones is not None and len(dones) > 0 and dones[0]:
            self.episode_rewards.append(sum(self._current_rewards))
            self.episode_lengths.append(len(self._current_rewards))

            final_distance = (
                self._current_distances[-1]
                if self._current_distances
                else float("nan")
            )
            self.episode_distances.append(final_distance)

            if self._current_distances:
                self.episode_initial_distances.append(self._current_distances[0])

            final_success = False
            if self._current_success_flags:
                final_success = bool(self._current_success_flags[-1])
            elif infos is not None and len(infos) > 0:
                info0 = infos[0]
                if "is_success" in info0:
                    final_success = bool(info0["is_success"])
                elif np.isfinite(final_distance) and final_distance < self.success_threshold:
                    final_success = True

            self.episode_successes.append(final_success)

            termination_reason = "none"
            expected_path_length = float("nan")
            actual_path_length = float("nan")
            if infos is not None and len(infos) > 0:
                termination_reason = str(infos[0].get("termination_reason", "none"))
                expected_path_length = float(infos[0].get("expected_path_length", float("nan")))
                actual_path_length = float(infos[0].get("actual_path_length", float("nan")))
            self.episode_termination_reasons.append(termination_reason)
            self.episode_expected_path_lengths.append(expected_path_length)
            self.episode_actual_path_lengths.append(actual_path_length)

            for key in REWARD_COMPONENT_KEYS:
                comp_list = getattr(self, f"_current_{key}")
                getattr(self, f"episode_{key}").append(sum(comp_list))

            self._current_rewards = []
            self._current_distances = []
            self._current_success_flags = []
            self._current_action_norms = []
            for key in REWARD_COMPONENT_KEYS:
                getattr(self, f"_current_{key}").clear()

            ep_idx = len(self.episode_rewards)
            if ep_idx % self.episode_log_interval == 0 and self.verbose:
                self._print_episode_summary(ep_idx)

        return True

    def _print_episode_summary(self, ep_idx: int) -> None:
        window = self.episode_log_interval
        recent_r = self.episode_rewards[-window:]
        recent_l = self.episode_lengths[-window:]
        recent_s = self.episode_successes[-window:]
        recent_d = self.episode_distances[-window:]
        recent_init_d = self.episode_initial_distances[-window:]
        recent_terms = self.episode_termination_reasons[-window:]
        recent_exp_pl = self.episode_expected_path_lengths[-window:]
        recent_act_pl = self.episode_actual_path_lengths[-window:]

        avg_r = _safe_mean(recent_r)
        avg_l = _safe_mean(recent_l, 0.0)
        avg_d = _safe_mean(recent_d)
        n_recent = len(recent_s)
        success_rate = sum(1 for s in recent_s if s) / n_recent if n_recent else 0.0

        collision_rate = sum(1 for t in recent_terms if t == "collision") / n_recent if n_recent else 0.0
        ws_violation_rate = sum(1 for t in recent_terms if t == "workspace_limit") / n_recent if n_recent else 0.0
        max_steps_rate = sum(1 for t in recent_terms if t == "max_steps") / n_recent if n_recent else 0.0

        valid_improvements = [
            init_d - final_d
            for init_d, final_d in zip(recent_init_d, recent_d)
            if np.isfinite(init_d) and np.isfinite(final_d)
        ]
        avg_improvement = _safe_mean(valid_improvements)

        success_exp_pl = [
            exp_pl for exp_pl, s in zip(recent_exp_pl, recent_s) if s and np.isfinite(exp_pl)
        ]
        success_act_pl = [
            act_pl for act_pl, s in zip(recent_act_pl, recent_s) if s and np.isfinite(act_pl)
        ]
        success_effs = [
            exp_pl / act_pl * 100.0
            for exp_pl, act_pl in zip(success_exp_pl, success_act_pl)
            if np.isfinite(exp_pl) and np.isfinite(act_pl) and act_pl > 1e-8
        ]

        avg_success_exp_pl = _safe_mean(success_exp_pl)
        avg_success_act_pl = _safe_mean(success_act_pl)
        avg_success_eff_pct = _safe_mean(success_effs)

        print(
            f"[EP {ep_idx:06d}] "
            f"reward={avg_r:8.3f} | "
            f"len={avg_l:5.0f} | "
            f"success={success_rate:.2f} | "
            f"collision={collision_rate:.2f} | "
            f"ws_viol={ws_violation_rate:.2f} | "
            f"max_steps={max_steps_rate:.2f} | "
            f"dist={avg_d:.4f} | "
            f"improve={avg_improvement:+.4f}"
        )

        print(
            f"  path eff%: "
            f"success_eff_pct={avg_success_eff_pct:6.2f}% | "
            f"success_exp_pl={avg_success_exp_pl:.4f} | "
            f"success_act_pl={avg_success_act_pl:.4f} | "
            f"(n_success={len(success_effs)})"
        )

        def _ep_avg(key: str) -> float:
            vals = getattr(self, f"episode_{key}")[-window:]
            return _safe_mean(vals, 0.0)

        print(
            f"  reward (avg/step): "
            f"distance={_ep_avg('distance'):+.3f} "
            f"time={_ep_avg('time'):+.3f} "
            f"shake={_ep_avg('shake'):+.3f} "
            f"success={_ep_avg('success'):+.1f} "
            f"collision={_ep_avg('collision'):+.1f} "
            f"workspace={_ep_avg('workspace'):+.1f} "
            f"episode={_ep_avg('episode'):+.1f}"
        )


class TerminationStatsCallback(sb3_cb.BaseCallback):
    """
    Tracks termination reason distribution during training.

    Logs rolling averages every ``log_interval`` episodes to SB3's CSV logger.
    """

    def __init__(
        self,
        log_interval: int = 100,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_interval = log_interval
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._pending_initial_distances: list[float] = []
        self._total_episodes = 0
        self._success_count = 0
        self._collision_count = 0
        self._ws_violation_count = 0
        self._max_steps_count = 0
        self._improvement_sum = 0.0
        self._improvement_count = 0
        self._action_norm_sum = 0.0
        self._action_norm_count = 0
        self._success_expected_path_lengths: list[float] = []
        self._success_actual_path_lengths: list[float] = []
        self._all_expected_path_lengths: list[float] = []
        self._all_actual_path_lengths: list[float] = []

    def _on_training_start(self) -> None:
        self._reset_episode_state()
        self._reward_component_sums: dict = {k: 0.0 for k in REWARD_COMPONENT_KEYS}
        self._reward_component_counts: dict = {k: 0 for k in REWARD_COMPONENT_KEYS}
        self._pending_reward_components: list = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        actions = self.locals.get("actions")

        if actions is not None and len(actions) > 0:
            action = actions[0] if actions.ndim == 1 else actions[0]
            norm = float(np.linalg.norm(np.asarray(action).flatten()))
            self._action_norm_sum += norm
            self._action_norm_count += 1

        if infos is not None and len(infos) > 0:
            info = infos[0]
            if "distance" in info:
                self._pending_initial_distances.append(float(info["distance"]))
            if "action_norm" in info:
                self._action_norm_sum += float(info["action_norm"])
                self._action_norm_count += 1
            if "reward_components" in info:
                self._pending_reward_components.append(info["reward_components"])

        if dones is not None and len(dones) > 0 and dones[0]:
            self._total_episodes += 1

            term_reason = "none"
            if infos is not None and len(infos) > 0:
                term_reason = str(infos[0].get("termination_reason", "none"))

            if term_reason == "success":
                self._success_count += 1
            elif term_reason == "collision":
                self._collision_count += 1
            elif term_reason == "workspace_limit":
                self._ws_violation_count += 1
            elif term_reason == "max_steps":
                self._max_steps_count += 1

            expected_pl = float("nan")
            actual_pl = float("nan")
            if infos is not None and len(infos) > 0:
                expected_pl = float(infos[0].get("expected_path_length", float("nan")))
                actual_pl = float(infos[0].get("actual_path_length", float("nan")))
            self._all_expected_path_lengths.append(expected_pl)
            self._all_actual_path_lengths.append(actual_pl)

            if term_reason == "success":
                self._success_expected_path_lengths.append(expected_pl)
                self._success_actual_path_lengths.append(actual_pl)

            if self._pending_initial_distances:
                final_dist = 0.0
                if infos is not None and len(infos) > 0:
                    final_dist = float(infos[0].get("distance", 0.0))
                init_dist = self._pending_initial_distances[0]
                if np.isfinite(init_dist) and np.isfinite(final_dist):
                    self._improvement_sum += init_dist - final_dist
                    self._improvement_count += 1

            self._pending_initial_distances = []

            if self._total_episodes % self.log_interval == 0 and self.verbose:
                self._log_stats()

        return True

    def _log_stats(self) -> None:
        n = self._total_episodes
        if n == 0:
            return

        success_rate = self._success_count / n
        collision_rate = self._collision_count / n
        ws_violation_rate = self._ws_violation_count / n
        max_steps_rate = self._max_steps_count / n
        avg_improvement = self._improvement_sum / max(1, self._improvement_count)
        avg_action_norm = self._action_norm_sum / max(1, self._action_norm_count)

        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/collision_rate", collision_rate)
        self.logger.record("rollout/workspace_violation_rate", ws_violation_rate)
        self.logger.record("rollout/max_steps_rate", max_steps_rate)
        self.logger.record("rollout/distance_improvement_mean", avg_improvement)
        self.logger.record("rollout/action_norm_mean", avg_action_norm)

        finite_exp = [x for x in self._all_expected_path_lengths if np.isfinite(x)]
        finite_act = [x for x in self._all_actual_path_lengths if np.isfinite(x)]
        self.logger.record("rollout/all_expected_path_length_mean", _safe_mean(finite_exp))
        self.logger.record("rollout/all_actual_path_length_mean", _safe_mean(finite_act))

        finite_s_exp = [x for x in self._success_expected_path_lengths if np.isfinite(x)]
        finite_s_act = [x for x in self._success_actual_path_lengths if np.isfinite(x)]
        if finite_s_exp and finite_s_act:
            effs = [exp / act * 100.0 for exp, act in zip(finite_s_exp, finite_s_act) if act > 1e-8]
            if effs:
                self.logger.record("rollout/success_path_efficiency_percent_mean", _safe_mean(effs))
            self.logger.record("rollout/success_expected_path_length_mean", _safe_mean(finite_s_exp))
            self.logger.record("rollout/success_actual_path_length_mean", _safe_mean(finite_s_act))

        if not hasattr(self, "_reward_component_sums"):
            self._reward_component_sums = {k: 0.0 for k in REWARD_COMPONENT_KEYS}
            self._reward_component_counts = {k: 0 for k in REWARD_COMPONENT_KEYS}
        if not hasattr(self, "_pending_reward_components"):
            self._pending_reward_components = []

        for rc in self._pending_reward_components:
            for key in REWARD_COMPONENT_KEYS:
                self._reward_component_sums[key] += rc.get(key, 0.0)
                self._reward_component_counts[key] += 1

        self._pending_reward_components.clear()

        for key in REWARD_COMPONENT_KEYS:
            avg_val = self._reward_component_sums[key] / max(1, self._reward_component_counts[key])
            self.logger.record(f"reward/{key}", avg_val)

        self.logger.record("rollout/reward_steps", max(1, sum(self._reward_component_counts.values())))
        self.logger.record("rollout/n_episodes", n)


class EarlyStopSafetyCallback(sb3_cb.BaseCallback):
    """
    Safety guard for fine-tuning and curriculum continuation runs.

    Monitors the rolling success rate over a sliding window of episodes.
    If success_rate falls below ``min_success_rate`` after the warmup period,
    training is halted to prevent policy collapse.

    Also monitors ``action_norm`` from infos.  If the rolling mean action norm
    exceeds ``max_action_norm``, it signals a collapse in progress.
    """

    def __init__(
        self,
        min_success_rate: float = 0.90,
        warmup_timesteps: int = 50000,
        warmup_episodes: int = 200,
        check_every_n_episodes: int = 20,
        check_window_size: int = 20,
        max_action_norm: float | None = 1.5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.min_success_rate = min_success_rate
        self.warmup_timesteps = warmup_timesteps
        self.warmup_episodes = warmup_episodes
        self.check_every_n_episodes = check_every_n_episodes
        self.check_window_size = check_window_size
        self.max_action_norm = max_action_norm

        self._successes: list[bool] = []
        self._action_norms: list[float] = []
        self._episodes_since_check = 0
        self._consecutive_failures = 0
        self._stopped = False
        self._warned_about_warmup = False

    def _in_warmup(self) -> bool:
        total_steps = self.num_timesteps
        total_episodes = len(self._successes)
        after_time = self.warmup_timesteps <= 0 or total_steps >= self.warmup_timesteps
        after_ep = self.warmup_episodes <= 0 or total_episodes >= self.warmup_episodes
        return not (after_time and after_ep)

    def _on_step(self) -> bool:
        if self._stopped:
            return False

        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if infos is None or dones is None:
            return True

        for done, info in zip(dones, infos):
            if bool(done):
                is_success = bool(info.get("is_success", False))
                self._successes.append(is_success)
                self._episodes_since_check += 1

            action_norm = info.get("action_norm")
            if action_norm is not None:
                self._action_norms.append(float(action_norm))

        if self._in_warmup() and not self._warned_about_warmup:
            if self.verbose:
                print(
                    f"[SAFE] Warmup ended @ step={self.num_timesteps:,}  "
                    f"episodes={len(self._successes):,}"
                )
            self._warned_about_warmup = True

        if self._episodes_since_check >= self.check_every_n_episodes:
            self._check_stability()
            self._episodes_since_check = 0

        return not self._stopped

    def _check_stability(self) -> None:
        in_warmup = self._in_warmup()
        n = min(len(self._successes), self.check_window_size)

        action_norm_ok = True
        action_norm_msg = ""
        if self.max_action_norm is not None and len(self._action_norms) >= self.check_window_size:
            recent_norms = self._action_norms[-self.check_window_size:]
            rolling_action_norm = sum(recent_norms) / len(recent_norms)
            action_norm_ok = rolling_action_norm < self.max_action_norm
            action_norm_msg = f" | rolling_action_norm={rolling_action_norm:.4f} (max={self.max_action_norm:.4f})"

        if not action_norm_ok:
            print(
                f"\n[SAFE] *** ACTION NORM EXCEEDED — STOPPING ***\n"
                f"  step={self.num_timesteps:,}  episodes={len(self._successes):,}\n"
            )
            self._stopped = True
            return

        if in_warmup:
            if n < self.check_window_size:
                return
            recent = self._successes[-self.check_window_size:]
            rolling_sr = sum(recent) / len(recent)
            if self.verbose:
                print(
                    f"[SAFE] warmup check @ ep {len(self._successes):d}: "
                    f"rolling_success_rate={rolling_sr:.3f} "
                    f"(threshold={self.min_success_rate:.3f} after warmup) | "
                    f"WARMUP{action_norm_msg}"
                )
            return

        if n < self.check_window_size:
            if self.verbose:
                print(
                    f"[SAFE] Not enough episodes ({len(self._successes)}) "
                    f"to compute rolling success rate (need {self.check_window_size})"
                )
            return

        recent = self._successes[-self.check_window_size:]
        rolling_success_rate = sum(recent) / len(recent)

        status = "OK" if rolling_success_rate >= self.min_success_rate else "FAIL"
        marker = "*** STOPPING ***" if rolling_success_rate < self.min_success_rate else ""

        if self.verbose:
            print(
                f"[SAFE] check @ ep {len(self._successes):d}: "
                f"rolling_success_rate={rolling_success_rate:.3f} "
                f"(min={self.min_success_rate:.3f}) | "
                f"status={status}{action_norm_msg} {marker}"
            )

        if rolling_success_rate < self.min_success_rate:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 1:
                print(
                    f"\n[SAFE] *** EARLY STOP triggered ***\n"
                    f"[SAFE]   step={self.num_timesteps:,}  episodes={len(self._successes):,}\n"
                    f"[SAFE]   rolling_success_rate={rolling_success_rate:.3f} "
                    f"(min={self.min_success_rate:.3f})\n"
                    f"[SAFE]   Halting training to prevent further policy collapse.\n"
                )
                self._stopped = True
        else:
            self._consecutive_failures = 0


class CurriculumGlobalStepCallback(sb3_cb.BaseCallback):
    """
    SB3 callback that keeps the master curriculum sampler in sync with the
    training global step and broadcasts stage transitions to all workers.

    It also tracks per-stage episode metrics and emits a summary report at
    the end of training.
    """

    def __init__(
        self,
        master_sampler: "CurriculumTargetSampler",
        worker_samplers: list,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self._master = master_sampler
        self._workers = worker_samplers

        self._stage_episodes: dict[str, list[dict]] = {}
        self._current_stage_name: str = master_sampler.current_stage.name
        self._reported_stage_names: set[str] = set()
        self._reported_stage_names.add(self._current_stage_name)

        self._episodes_this_check = 0

    def on_env_reset(self, env: "CartesianPathPlanningEnv") -> None:
        """Called by ``CartesianPathPlanningEnv`` at the end of every reset()."""
        self._master.on_episode_end()

        new_stage_name = self._master.current_stage.name
        if new_stage_name != self._current_stage_name:
            self._on_stage_change(new_stage_name)

        for worker in self._workers:
            worker.sync_from(self._master)

    def _on_stage_change(self, new_stage_name: str) -> None:
        old_stage = self._current_stage_name
        self._current_stage_name = new_stage_name
        self._reported_stage_names.add(new_stage_name)

        if self.verbose:
            print(
                f"\n"
                f"============================================================\n"
                f"[CURRICULUM] *** STAGE TRANSITION ***\n"
                f"  From: {old_stage}\n"
                f"  To:   {new_stage_name}\n"
                f"  Mode: {self._master.current_stage.mode}\n"
                f"  Jitter radius: {self._master.current_stage.jitter_radius:.3f} m\n"
                f"  Stage timesteps: {self._master.current_stage.timesteps:,}\n"
                f"  Total episodes so far: {self._master._episode_count:,}\n"
                f"============================================================\n"
            )

    def _track_episode(self, info: dict) -> None:
        stage_name = self._current_stage_name
        if stage_name not in self._stage_episodes:
            self._stage_episodes[stage_name] = []

        record = {
            "is_success": bool(info.get("is_success", False)),
            "final_distance": float(info.get("distance", float("inf"))),
            "step_count": int(info.get("step_count", 0)),
            "path_length": float(info.get("path_length_so_far", 0.0)),
            "action_delta": float(info.get("action_delta", 0.0)),
            "path_efficiency": float(info.get("path_efficiency_so_far", 0.0)),
        }
        self._stage_episodes[stage_name].append(record)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is not None and dones is not None:
            for done, info in zip(dones, infos):
                if bool(done):
                    self._episodes_this_check += 1
                    self._track_episode(info)

        for worker in self._workers:
            worker.sync_from(self._master)

        return True

    def _on_training_end(self) -> None:
        self._print_stage_report()

    def _print_stage_report(self) -> None:
        print(
            f"\n"
            f"================================================================\n"
            f"[CURRICULUM] STAGE REPORT (end of training)\n"
            f"================================================================"
        )

        all_stages = self._master.config.stages
        for stage in all_stages:
            sname = stage.name
            episodes = self._stage_episodes.get(sname, [])
            n = len(episodes)

            if n == 0:
                print(f"\n  Stage: {sname} ({stage.mode})")
                print(f"    Episodes: 0 — no data (stage may not have been reached)")
                continue

            successes = [e for e in episodes if e["is_success"]]
            success_rate = len(successes) / n

            final_distances = [e["final_distance"] for e in episodes]
            mean_final_dist = _safe_mean(final_distances)

            path_lengths = [e["path_length"] for e in episodes]
            mean_path_length = _safe_mean(path_lengths)

            efficiencies = [e["path_efficiency"] for e in episodes]
            mean_efficiency = _safe_mean(efficiencies)

            action_deltas = [e["action_delta"] for e in episodes]
            mean_action_delta = _safe_mean(action_deltas)

            print(
                f"\n  Stage: {sname}\n"
                f"    Mode: {stage.mode}\n"
                f"    Description: {stage.description}\n"
                f"    Jitter radius: {stage.jitter_radius:.3f} m\n"
                f"    Timesteps: {stage.timesteps:,}\n"
                f"    Episodes: {n}\n"
                f"    Success rate: {success_rate*100:.1f}%\n"
                f"    Mean final distance: {mean_final_dist:.4f} m\n"
                f"    Mean path length: {mean_path_length:.4f} m\n"
                f"    Mean path efficiency: {mean_efficiency:.4f}\n"
                f"    Mean action delta: {mean_action_delta:.4f}"
            )

        print(
            f"\n"
            f"================================================================\n"
            f"[CURRICULUM] END OF REPORT\n"
            f"================================================================\n"
        )


# -------------------------------------------------------------------------- #
# Checkpoint saving callbacks
# -------------------------------------------------------------------------- #


class EvalBestModelCallback(sb3_cb.BaseCallback):
    """
    Evaluate the model at regular intervals and save it when eval success rate improves.

    Runs ``n_eval_episodes`` episodes in a deterministic eval environment
    (created from the same Config) and compares the result against the best
    seen so far using the priority::

        1. Higher ``eval_success_rate`` wins.
        2. Ties broken by higher ``eval_mean_reward``.
        3. Further ties broken by lower ``eval_mean_final_distance``.

    Saves to ``run_dir/model/best_model.zip`` plus companion metadata
    ``run_dir/model/best_model_info.json``.

    Callbacks that should **not** appear in the same callback list as this one
    (they would duplicate work):

        - ``stable_baselines3.common.callbacks.EvalCallback``
        - ``SaveBestModelCallback`` (the old rolling-training version)

    Parameters
    ----------
    eval_freq
        Run eval every this many timesteps.  Set to ``cfg.training.eval_freq``.
    n_eval_episodes
        Number of episodes per eval run.  Set to ``cfg.evaluation.num_episodes``.
    eval_env_cfg
        The full ``Config`` object used to create the training environment.
        The eval environment is created identically (same start/target distributions).
    eval_seed
        Random seed for the eval environment.
    model_dir
        Directory to write ``best_model.zip`` and ``best_model_info.json`` into.
    warmup_timesteps
        Ignore the first ``warmup_timesteps`` timesteps before starting to eval.
    warmup_episodes
        Wait for at least this many total training episodes before saving any model.
    verbose
        Print a line each time a new best model is found.
    """

    def __init__(
        self,
        eval_freq: int,
        n_eval_episodes: int,
        eval_env_cfg: "drl_pathplanning.gymnasium.config.Config",
        eval_seed: int,
        model_dir: Path,
        warmup_timesteps: int = 0,
        warmup_episodes: int = 0,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_env_cfg = eval_env_cfg
        self.eval_seed = eval_seed
        self.model_dir = Path(model_dir)
        self.warmup_timesteps = warmup_timesteps
        self.warmup_episodes = warmup_episodes
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._total_episodes = 0
        self._last_eval_step = -1
        self._best_sr = -1.0
        self._best_reward = float("-inf")
        self._best_distance = float("inf")
        self._eval_env: "gym.Env | None" = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_eval_env(self) -> "gym.Env":
        """Lazily create the single deterministic eval environment."""
        if self._eval_env is not None:
            return self._eval_env

        from drl_pathplanning.gymnasium import CartesianPathPlanningEnv
        env = CartesianPathPlanningEnv(
            env_cfg=self.eval_env_cfg,
        )
        env.reset(seed=self.eval_seed)
        self._eval_env = env
        return env

    def _run_eval(self) -> dict:
        """
        Run ``n_eval_episodes`` in the eval env with deterministic policy.

        Returns a dict with keys: ``success_rate``, ``mean_reward``,
        ``mean_final_distance``, ``collision_rate``, ``ws_viol_rate``.
        """
        import math

        env = self._get_eval_env()
        successes: list[bool] = []
        rewards: list[float] = []
        distances: list[float] = []
        collisions = 0
        ws_viols = 0

        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

            successes.append(bool(info.get("is_success", False)))
            rewards.append(ep_reward)
            dist = float(info.get("distance", float("nan")))
            distances.append(dist)
            if info.get("termination_reason") == "collision":
                collisions += 1
            elif info.get("termination_reason") in ("workspace_limit", "out_of_workspace"):
                ws_viols += 1

        n = self.n_eval_episodes
        finite_distances = [d for d in distances if not math.isnan(d)]
        mean_dist = sum(finite_distances) / len(finite_distances) if finite_distances else float("nan")
        return {
            "success_rate": sum(successes) / n,
            "mean_reward": sum(rewards) / n,
            "mean_final_distance": mean_dist,
            "mean_final_distance_raw": distances,
            "collision_rate": collisions / n,
            "ws_viol_rate": ws_viols / n,
        }

    def _is_better(self, new: dict, best: tuple) -> bool:
        """Compare new eval result against (best_sr, best_reward, best_distance)."""
        import math
        new_sr, new_rew = new["success_rate"], new["mean_reward"]
        new_dist_raw = new["mean_final_distance"]
        new_dist = new_dist_raw if not math.isnan(new_dist_raw) else float("inf")

        best_sr, best_rew, best_dist_raw = best
        best_dist = best_dist_raw if not math.isnan(best_dist_raw) else float("inf")

        if new_sr > best_sr:
            return True
        if new_sr == best_sr:
            if new_rew > best_rew:
                return True
            if new_rew == best_rew:
                return new_dist < best_dist
        return False

    # ------------------------------------------------------------------ #
    # SB3 callback hook
    # ------------------------------------------------------------------ #

    def _on_step(self) -> bool:
        ts = self.num_timesteps

        # Count episode boundaries via infos
        infos: list[dict] = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                self._total_episodes += 1

        # Skip until warmup thresholds are met
        if ts < self.warmup_timesteps:
            return True
        if self._total_episodes < self.warmup_episodes:
            return True

        # Only eval at the configured interval
        if ts - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = ts

        # Run eval
        result = self._run_eval()

        is_better = self._is_better(result, (self._best_sr, self._best_reward, self._best_distance))
        if is_better:
            self._best_sr = result["success_rate"]
            self._best_reward = result["mean_reward"]
            self._best_distance = result["mean_final_distance"]

            best_path = self.model_dir / "best_model.zip"
            self.model.save(str(best_path))

            import json
            from datetime import datetime, timezone
            info_path = self.model_dir / "best_model_info.json"

            def _nan_to_none(v):
                """Convert NaN to None for JSON compatibility; leave finite floats unchanged."""
                import math
                if isinstance(v, float) and math.isnan(v):
                    return None
                return v

            meta = {
                "step": int(ts),
                "eval_success_rate": float(result["success_rate"]),
                "eval_mean_reward": float(result["mean_reward"]),
                "eval_mean_final_distance": _nan_to_none(result["mean_final_distance"]),
                "eval_collision_rate": float(result["collision_rate"]),
                "eval_workspace_violation_rate": float(result["ws_viol_rate"]),
                "eval_episodes": int(self.n_eval_episodes),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(info_path, "w") as fh:
                json.dump(meta, fh, indent=4)

            if self.verbose:
                import math
                dist_val = result["mean_final_distance"]
                dist_str = f"{dist_val:.4f}" if not math.isnan(dist_val) else "N/A"
                print(
                    f"[BEST] New best model saved\n"
                    f"  step={ts:,}  success_rate={result['success_rate']:.4f}  "
                    f"mean_reward={result['mean_reward']:.2f}  "
                    f"mean_final_distance={dist_str}"
                )

        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None


class SaveBestModelCallback(sb3_cb.BaseCallback):
    """
    DEPRECATED — prefer ``EvalBestModelCallback`` which uses proper evaluation.

    This callback tracks rolling training-episode success rates and saves
    ``best_checkpoint_t{timestep}.zip`` snapshots when the rolling success
    rate peaks, plus a companion ``best_model.zip``.  The ``best_model.zip``
    saved by this callback reflects training performance, not evaluation
    performance, and is kept only for backward compatibility with existing
    ``find_best_checkpoint.py`` tooling.

    Use ``EvalBestModelCallback`` for thesis-quality model selection.
    """

    def __init__(
        self,
        model_dir: Path,
        window: int = 50,
        min_episodes: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.model_dir = Path(model_dir)
        self.window = window
        self.min_episodes = min_episodes
        self._is_success_history: list[bool] = []
        self._best_sr = -1.0
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos: list[dict] = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "episode" not in info:
                continue
            ep = info["episode"]
            success = bool(
                ep.get("is_success", False)
                or ep.get("is_success", 0)
                or info.get("is_success", False)
                or info.get("success", False)
            )
            self._is_success_history.append(success)
            if len(self._is_success_history) > self.window:
                self._is_success_history.pop(0)

        if len(self._is_success_history) < self.min_episodes:
            return True

        rolling_sr = sum(self._is_success_history) / len(self._is_success_history)
        if rolling_sr > self._best_sr:
            self._best_sr = rolling_sr
            ts = self.num_timesteps
            best_ckpt = self.model_dir / f"best_checkpoint_t{ts}.zip"
            self.model.save(str(best_ckpt))
            best_overall = self.model_dir / "best_model.zip"
            self.model.save(str(best_overall))
            if self.verbose:
                print(
                    f"[SaveBestModel] NEW BEST @ step={ts:,} | "
                    f"rolling_success_rate={rolling_sr:.4f} | "
                    f"file={best_ckpt.name}"
                )
        return True


class SaveCheckpointCallback(sb3_cb.BaseCallback):
    """
    Save a model checkpoint every N timesteps.

    Saves ``run_dir/model/checkpoint_t{timestep}.zip`` on every ``save_freq``
    timesteps.  Does not overwrite previous checkpoints, creating a timeline
    of snapshots that can be compared with ``find_best_checkpoint.py``.

    Parameters
    ----------
    model_dir
        Directory to write checkpoint files into.
    save_freq
        Save every this many timesteps.
    verbose
        Print a line when saving.
    """

    def __init__(
        self,
        model_dir: Path,
        save_freq: int = 50000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.model_dir = Path(model_dir)
        self.save_freq = save_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._last_save = self.num_timesteps
            ckpt_path = self.model_dir / f"checkpoint_t{self._last_save}.zip"
            self.model.save(str(ckpt_path))
            if self.verbose:
                print(f"[SaveCheckpoint] step={self._last_save:,} -> {ckpt_path.name}")
        return True
