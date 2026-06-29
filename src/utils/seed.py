"""Reproducibility helper: seed all RNGs used during training/eval."""

import logging
import os
import random

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and Torch RNGs for reproducible runs.

    Args:
        seed: The seed value applied to every RNG.
        deterministic: If True, also request deterministic cuDNN/torch
            algorithms. This can slow training slightly and a few ops have no
            deterministic implementation, so failures are downgraded to a
            warning rather than raised.

    Returns:
        The seed that was applied (for logging/recording in run metadata).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep in practice, but stay defensive
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:  # pragma: no cover - backend dependent
                logger.warning("Could not enable deterministic algorithms: %s", exc)
    except ImportError:
        pass

    logger.info("Seeded all RNGs with seed=%d (deterministic=%s)", seed, deterministic)
    return seed
