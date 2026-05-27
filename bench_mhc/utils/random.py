"""Helper functions to fix random generator seeds."""

import polars as pl
from lightning.pytorch import seed_everything

from bench_mhc.utils.logging import system

log = system.get(__name__)


def set_random_seed(seed: int | None) -> int:
    """Set the random seed for reproducibility of our experiments.

    Setting this random seed will notably enable reproducibility regarding:
    - data preparation, e.g. shuffled batches
    - data over/under sampling
    - weight initialization

    Args:
        seed: the seed to set. If not provided, a random one is selected.
    """
    seed = seed_everything(seed, workers=True)
    pl.set_random_seed(seed)

    log.info(f"The random seed has been set to {seed}.")

    return seed
