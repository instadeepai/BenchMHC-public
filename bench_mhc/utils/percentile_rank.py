"""Utility module for computing and saving percentile ranks."""

from pathlib import Path

import numpy as np


class PercentileRankCalculator:
    """Class for computing and saving percentile ranks.

    This class provides functionality to calculate percentile ranks from a set of prediction scores.
    It divides the score range into a specified number of steps and computes bin edges at each
    percentile step. The bin edges can later be used to map new prediction scores to their
    corresponding percentile ranks.

    The number of steps defaults to NUM_STEPS_PCTRNK (10,000) which provides fine-grained
    percentile rank calculations.

    Attributes:
        steps: Array of percentile values from 0 to 100.
        bin_edges: Array of bin edges corresponding to each percentile step. Each value represents
            the score threshold at that percentile.
    """

    def __init__(self, num_steps: int | None = None, pctrnk_path: Path | None = None) -> None:
        """Initialize the calculator.

        Args:
            num_steps: Number of steps in the percentile rank calculation.
            pctrnk_path: Path to the percentile rank file. If provided, the percentile ranks
                will be loaded from the file.

        Raises:
            ValueError: If num_steps is not provided and pctrnk_path is not provided.
        """
        if pctrnk_path is not None:
            percentile_ranks = np.load(pctrnk_path)
            self.steps = percentile_ranks["steps"]
            self.bin_edges = percentile_ranks["bin_edges"]

        else:
            if num_steps is None:
                raise ValueError("num_steps must be provided if pctrnk_path is not provided")

            self.steps = np.linspace(0, 100, num_steps + 1)
            self.bin_edges = None

    def compute(self, scores: np.ndarray) -> dict[str, np.ndarray]:
        """Compute percentile ranks for a set of scores.

        Args:
            scores: Array of prediction scores.

        Returns:
            Dictionary containing steps and bin edges.
        """
        self.bin_edges = np.percentile(scores, self.steps, method="median_unbiased")
        self.bin_edges[0] = 0
        self.bin_edges[-1] = 1

        return {"steps": self.steps, "bin_edges": self.bin_edges}

    def convert_probs_to_percentile_ranks(self, probs: np.ndarray) -> np.ndarray:
        """Convert probabilities to percentile ranks.

        Args:
            probs: Array of probabilities.

        Returns:
            Array of percentile ranks.

        Raises:
            ValueError: If percentile ranks have not been computed yet.
        """
        if self.bin_edges is None:
            raise ValueError("Percentile ranks have not been computed yet.")

        return 100 - np.interp(x=probs, xp=self.bin_edges, fp=self.steps)
