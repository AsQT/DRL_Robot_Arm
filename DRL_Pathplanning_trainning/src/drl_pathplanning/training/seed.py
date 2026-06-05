"""
Seed utilities for reproducible DRL training.

Seeds all random sources used by the project:
  - Python built-in random
  - NumPy
  - PyTorch (CPU + CUDA)
  - Gymnasium environment seeder
"""

from __future__ import annotations

import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed all deterministic components of the training pipeline.

    Parameters
    ----------
    seed
        Integer seed value.  Using the same seed across runs guarantees
        identical behaviour (e.g. identical random target sampling).
    """
    # Python built-in.
    random.seed(seed)

    # NumPy.
    np.random.seed(seed)

    # PyTorch — CPU.
    torch.manual_seed(seed)

    # PyTorch — CUDA (each GPU gets the same seed).
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Disable nondeterministic cuDNN algorithms for fully reproducible GPU runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_numpy_generator(seed: int) -> np.random.Generator:
    """
    Return a NumPy Generator seeded with ``seed``.

    Use this instead of ``np_random`` when you need explicit control over
    the random stream (e.g. for per-run split generators).

    Parameters
    ----------
    seed
        Integer seed for the generator.

    Returns
    -------
    np.random.Generator
    """
    return np.random.default_rng(seed)
