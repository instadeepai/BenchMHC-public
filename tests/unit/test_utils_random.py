"""Unit tests related to bench_mhc/utils/random.py."""

import contextlib
import os
from collections.abc import Iterator
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from lightning import seed_everything

from bench_mhc.utils.random import set_random_seed


@contextlib.contextmanager
def pl_global_seed_ctx() -> Iterator[None]:
    """Context manager to temporarily remove PL_GLOBAL_SEED on which Lightning relies."""
    old_environ = dict(os.environ)
    os.environ.pop("PL_GLOBAL_SEED", None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@pytest.mark.parametrize("seed", [None, 42])
@patch("bench_mhc.utils.random.seed_everything", wraps=seed_everything)
@patch("bench_mhc.utils.random.pl.set_random_seed")
def test_set_random_seed(
    set_random_seed_mock: MagicMock,
    seed_everything_mock: MagicMock,
    seed: int | None,
) -> None:
    """Check set_random_seed works as expected."""
    with pl_global_seed_ctx():
        seed_set = set_random_seed(seed)

    expected_seed = seed if seed is not None else 0

    seed_everything_mock.assert_called_once_with(seed, workers=True)
    set_random_seed_mock.assert_called_once_with(expected_seed)
    assert seed_set == expected_seed
