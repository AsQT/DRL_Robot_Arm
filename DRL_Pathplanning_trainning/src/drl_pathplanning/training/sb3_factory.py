"""
Stable-Baselines3 model factory.

Provides ``create_sb3_model()`` which builds a SB3 model (DDPG / SAC / TD3 / PPO)
from an algorithm config dict and a (possibly wrapped) environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    import stable_baselines3


def create_sb3_model(
    algorithm: str,
    env: "Any",
    algo_cfg: dict[str, Any],
    device: str = "auto",
) -> "stable_baselines3.BaseAlgorithm":
    """
    Create and return a Stable-Baselines3 model.

    Parameters
    ----------
    algorithm
        One of ``"DDPG"``, ``"SAC"``, ``"TD3"``.
    env
        The (possibly wrapped) environment to train on.
    algo_cfg
        Parsed algorithm YAML config dict.  Expected keys:

        - ``policy`` — SB3 policy class name (default: ``"MlpPolicy"``).
        - ``gamma`` — discount factor (default: ``0.99``).
        - ``learning_rate`` — optimizer learning rate (default: ``3e-4``).
        - ``batch_size`` — training batch size (default: ``256``).
        - ``buffer_size`` — replay buffer size (default: ``100000``).
        - ``learning_starts`` — steps before training begins (default: ``100``).
        - ``tau`` — soft update coefficient (default: ``0.005``).
        - ``train_freq`` — environment steps between training updates (default: ``1``).
        - ``gradient_steps`` — gradient updates per training call (default: ``1``).
        - ``policy_kwargs.net_arch`` — neural-net hidden layers (default: ``[256,256]``).
        - ``action_noise`` — sub-dict with ``type``, ``mean``, ``sigma``; for DDPG/TD3.
        - ``target_policy_noise`` — noise added to target policy (TD3 top-level, default: ``0.2``).
        - ``target_noise_clip`` — target noise clipping range (TD3 top-level, default: ``0.5``).
        - ``policy_delay`` — policy update delay in steps (TD3 top-level, default: ``2``).
        - ``policy_noise`` — alias for ``target_policy_noise`` (for backward compat).
        - ``noise_clip`` — alias for ``target_noise_clip`` (for backward compat).
        - ``target_policy_delay`` — alias for ``policy_delay`` (for backward compat).
        - ``ent_coef`` — entropy coefficient (SAC, default: ``"auto"``).

    device
        Device string passed to SB3 (default: ``"auto"`` = CUDA if available).

    Returns
    -------
    stable_baselines3.BaseAlgorithm
        Instantiated SB3 model with logger set.

    Raises
    ------
    ValueError
        If ``algorithm`` is not supported.
    NotImplementedError
        If an unknown noise type is requested.
    """
    import stable_baselines3

    algo_upper = algorithm.upper()
    if algo_upper not in ("DDPG", "SAC", "TD3", "PPO"):
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Supported: DDPG, SAC, TD3, PPO"
        )
    algo_cls = getattr(stable_baselines3, algo_upper)

    # ── Hyperparameters ────────────────────────────────────────────────────
    gamma = float(algo_cfg.get("gamma", 0.99))
    learning_rate = float(algo_cfg.get("learning_rate", 3e-4))
    batch_size = int(algo_cfg.get("batch_size", 256))
    buffer_size = int(algo_cfg.get("buffer_size", 100_000))
    learning_starts = int(algo_cfg.get("learning_starts", 100))
    tau = float(algo_cfg.get("tau", 0.005))
    train_freq_raw = algo_cfg.get("train_freq", 1)
    if isinstance(train_freq_raw, dict):
        train_freq = train_freq_raw
    else:
        train_freq = int(train_freq_raw)
    gradient_steps = int(algo_cfg.get("gradient_steps", 1))

    policy = algo_cfg.get("policy", "MlpPolicy")
    net_arch = algo_cfg.get("policy_kwargs", {}).get("net_arch", [256, 256])
    policy_kwargs = dict(net_arch=net_arch)

    # ── Action noise (DDPG / TD3) ─────────────────────────────────────────
    action_noise = None
    if algo_upper in ("DDPG", "TD3"):
        noise_cfg = algo_cfg.get("action_noise", {})
        if noise_cfg:
            noise_type = noise_cfg.get("type", "NormalActionNoise")
            noise_sigma = float(noise_cfg.get("sigma", 0.1))
            if noise_type == "NormalActionNoise":
                n_action = env.action_space.shape[-1]
                action_noise = stable_baselines3.common.noise.NormalActionNoise(
                    mean=np.zeros(n_action),
                    sigma=noise_sigma * np.ones(n_action),
                )
            else:
                raise NotImplementedError(
                    f"Action noise type '{noise_type}' is not implemented. "
                    "Supported: NormalActionNoise"
                )

    # ── TD3-specific ────────────────────────────────────────────────────────
    # Note: policy_delay, target_policy_noise, target_noise_clip are top-level
    # TD3.__init__() parameters.  They are NOT policy_kwargs entries.
    if algo_upper == "TD3":
        target_policy_noise = float(algo_cfg.get(
            "target_policy_noise", algo_cfg.get("policy_noise", 0.2)
        ))
        target_noise_clip = float(algo_cfg.get(
            "target_noise_clip", algo_cfg.get("noise_clip", 0.5)
        ))
        policy_delay = int(algo_cfg.get(
            "policy_delay", algo_cfg.get("target_policy_delay", 2)
        ))

    # ── SAC-specific ────────────────────────────────────────────────────────
    if algo_upper == "SAC":
        ent_coef_raw = algo_cfg.get("ent_coef", "auto")
        if ent_coef_raw == "auto":
            ent_coef: "str | float" = "auto"
        else:
            ent_coef = float(ent_coef_raw)

    # ── Build model ──────────────────────────────────────────────────────────
    if algo_upper == "PPO":
        ppo_kwargs = dict(
            policy=policy,
            env=env,
            gamma=gamma,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=int(algo_cfg.get("n_epochs", 10)),
            n_steps=int(algo_cfg.get("n_steps", 2048)),
            gae_lambda=float(algo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(algo_cfg.get("clip_range", 0.2)),
            ent_coef=float(algo_cfg.get("ent_coef", 0.0)),
            vf_coef=float(algo_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(algo_cfg.get("max_grad_norm", 0.5)),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
        )
        model: "stable_baselines3.BaseAlgorithm" = algo_cls(**ppo_kwargs)
        return model

    common_kwargs: dict[str, Any] = dict(
        policy=policy,
        env=env,
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
    )

    if algo_upper in ("DDPG", "TD3"):
        common_kwargs["action_noise"] = action_noise

    if algo_upper == "TD3":
        common_kwargs["policy_delay"] = policy_delay
        common_kwargs["target_policy_noise"] = target_policy_noise
        common_kwargs["target_noise_clip"] = target_noise_clip

    if algo_upper == "SAC":
        common_kwargs["ent_coef"] = ent_coef

    model = algo_cls(**common_kwargs)
    return model
