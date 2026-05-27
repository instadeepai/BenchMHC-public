"""Unit tests related to bench_mhc/utils/percentile_rank.py."""

from pathlib import Path

import numpy as np
import pytest

from bench_mhc.utils.percentile_rank import PercentileRankCalculator


@pytest.fixture
def scores() -> np.ndarray:
    """Create a sample array of scores for testing."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


@pytest.fixture
def calculator(scores: np.ndarray) -> PercentileRankCalculator:
    """Create a PercentileRankCalculator instance with computed percentile ranks."""
    calculator = PercentileRankCalculator(num_steps=10)
    calculator.compute(scores)

    return calculator


def test_initialization() -> None:
    """Test initialization with custom number of steps."""
    num_steps = 100
    calculator = PercentileRankCalculator(num_steps=num_steps)
    assert len(calculator.steps) == num_steps + 1
    assert calculator.bin_edges is None


def test_initialization_no_num_steps_no_pctrnk_path() -> None:
    """Test initialization with no number of steps and no percentile rank path."""
    with pytest.raises(
        ValueError, match="num_steps must be provided if pctrnk_path is not provided"
    ):
        PercentileRankCalculator()


def test_initialization_from_file(scores: np.ndarray, tmp_path: Path) -> None:
    """Test initialization from a saved percentile rank file."""
    # Create a temporary file with percentile ranks
    calculator = PercentileRankCalculator(num_steps=10)
    percentile_ranks = calculator.compute(scores)

    path_to_percentile_ranks = tmp_path / "percentile_ranks.npz"
    np.savez(path_to_percentile_ranks, **percentile_ranks)

    loaded_calculator = PercentileRankCalculator(num_steps=10, pctrnk_path=path_to_percentile_ranks)
    assert np.array_equal(loaded_calculator.steps, calculator.steps)
    assert np.array_equal(loaded_calculator.bin_edges, calculator.bin_edges)


def test_compute(scores: np.ndarray) -> None:
    """Test computation of percentile ranks."""
    calculator = PercentileRankCalculator(num_steps=10)
    result = calculator.compute(scores)

    assert "steps" in result
    assert "bin_edges" in result
    assert len(result["steps"]) == len(result["bin_edges"])
    assert result["bin_edges"][0] == 0
    assert result["bin_edges"][-1] == 1
    assert calculator.bin_edges is not None
    assert np.array_equal(calculator.bin_edges, result["bin_edges"])


def test_convert_probs_to_percentile_ranks(calculator: PercentileRankCalculator) -> None:
    """Test conversion of probabilities to percentile ranks."""
    probs = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
    percentile_ranks = calculator.convert_probs_to_percentile_ranks(probs)

    assert len(percentile_ranks) == len(probs)
    assert np.all(percentile_ranks >= 0)
    assert np.all(percentile_ranks <= 100)
    # Test that adjacent differences are non-positive since
    # Higher probabilities should correspond to lower percentile ranks
    assert np.all(np.diff(percentile_ranks) <= 0)


def test_convert_probs_to_percentile_ranks_uninitialized() -> None:
    """Test error when converting probabilities without computing percentile ranks."""
    calculator = PercentileRankCalculator(num_steps=1000)
    with pytest.raises(ValueError, match="Percentile ranks have not been computed yet"):
        calculator.convert_probs_to_percentile_ranks(np.array([0.5]))
