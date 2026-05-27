# Expr.binary_metrics is added at runtime
# mypy: disable-error-code="attr-defined"
"""Unit test cases for custom polars namespace `binary_metrics`."""

from typing import Any

import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


@pytest.mark.parametrize("use_sample_weight", [True, False])
def test_pl_binary_metrics(use_sample_weight: bool) -> None:
    """Ensure the metrics defined in "binary_metrics" namespace give same results as sklearn.

    Both global and per allele metrics are checked.
    """

    def _evaluate_sklearn(data: pl.DataFrame) -> dict[str, float]:
        """Evaluate metrics using sklearn.

        Args:
            data: Dataframe with true labels, predicted probabilities and predicted labels.

        Returns:
            Dictionary with metrics.
        """
        y_true = data["y_true"].to_numpy()
        y_proba = data["y_proba"].to_numpy()
        y_pred = data["y_pred"].to_numpy()

        sample_weight = None
        if "sample_weight" in data.columns:
            sample_weight = data["sample_weight"].to_numpy()

        precision_, recall_, _ = precision_recall_curve(
            y_true, y_proba, sample_weight=sample_weight
        )

        return {
            "roc_auc": roc_auc_score(y_true, y_proba, sample_weight=sample_weight),
            "average_precision": average_precision_score(
                y_true, y_proba, sample_weight=sample_weight
            ),
            "f1": f1_score(y_true, y_pred, sample_weight=sample_weight),
            "recall": recall_score(y_true, y_pred, sample_weight=sample_weight),
            "precision": precision_score(y_true, y_pred, sample_weight=sample_weight),
            "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred, sample_weight=sample_weight
            ),
            "pr_auc": auc(recall_, precision_),
            "fbeta": fbeta_score(y_true, y_pred, beta=2, sample_weight=sample_weight),
        }

    num_samples = 100_000

    num_alleles = 10

    allele_candidates = [f"HLA-{i}" for i in range(num_alleles)]

    df = pl.DataFrame(
        {
            "y_true": (np.random.rand(num_samples) > 0.99).astype(int),
            "y_proba": np.random.rand(num_samples),
            "allele": np.random.choice(allele_candidates, size=num_samples),
        }
    ).with_columns(
        # to compute precision, recall, ... we need the predicted labels (same as sklearn)
        (pl.col("y_proba") > 0.5).cast(pl.Int32).alias("y_pred")
    )

    if use_sample_weight:
        df = df.with_columns((pl.lit(np.random.rand(df.height))).alias("sample_weight"))

        metrics_pl_expr = [
            pl.col("y_true")
            .binary_metrics.precision_score(pl.col("y_pred"), pl.col("sample_weight"))
            .alias("precision"),
            pl.col("y_true")
            .binary_metrics.recall_score(pl.col("y_pred"), pl.col("sample_weight"))
            .alias("recall"),
            pl.col("y_true")
            .binary_metrics.f1_score(pl.col("y_pred"), pl.col("sample_weight"))
            .alias("f1"),
            pl.col("y_true")
            .binary_metrics.accuracy_score(pl.col("y_pred"), pl.col("sample_weight"))
            .alias("accuracy"),
            pl.col("y_true")
            .binary_metrics.balanced_accuracy_score(pl.col("y_pred"), pl.col("sample_weight"))
            .alias("balanced_accuracy"),
            pl.col("y_true")
            .binary_metrics.average_precision_score(pl.col("y_proba"), pl.col("sample_weight"))
            .alias("average_precision"),
            pl.col("y_true")
            .binary_metrics.roc_auc_score(pl.col("y_proba"), pl.col("sample_weight"))
            .alias("roc_auc"),
            pl.col("y_true")
            .binary_metrics.pr_auc_score(pl.col("y_proba"), pl.col("sample_weight"))
            .alias("pr_auc"),
            pl.col("y_true")
            .binary_metrics.fbeta_score(
                pl.col("y_pred"), beta=2, sample_weight=pl.col("sample_weight")
            )
            .alias("fbeta"),
        ]

    else:
        metrics_pl_expr = [
            pl.col("y_true").binary_metrics.precision_score(pl.col("y_pred")).alias("precision"),
            pl.col("y_true").binary_metrics.recall_score(pl.col("y_pred")).alias("recall"),
            pl.col("y_true").binary_metrics.f1_score(pl.col("y_pred")).alias("f1"),
            pl.col("y_true").binary_metrics.accuracy_score(pl.col("y_pred")).alias("accuracy"),
            pl.col("y_true")
            .binary_metrics.balanced_accuracy_score(pl.col("y_pred"))
            .alias("balanced_accuracy"),
            pl.col("y_true")
            .binary_metrics.average_precision_score(pl.col("y_proba"))
            .alias("average_precision"),
            pl.col("y_true").binary_metrics.roc_auc_score(pl.col("y_proba")).alias("roc_auc"),
            pl.col("y_true").binary_metrics.pr_auc_score(pl.col("y_proba")).alias("pr_auc"),
            pl.col("y_true").binary_metrics.fbeta_score(pl.col("y_pred"), beta=2).alias("fbeta"),
        ]

    # Global metrics
    df_metrics_from_sklearn = pl.DataFrame(_evaluate_sklearn(df))

    df_metrics_from_pl = df.lazy().select(metrics_pl_expr).collect()

    pl_testing.assert_frame_equal(
        df_metrics_from_pl, df_metrics_from_sklearn, check_column_order=False
    )

    # Metrics per allele
    allele2sklearn_metrics: dict[Any, dict[Any, Any]] = {
        allele: _evaluate_sklearn(df.filter(pl.col("allele") == allele))
        for allele in df.get_column("allele").unique().sort().to_list()
    }
    df_metrics_from_sklearn = (
        pl.DataFrame(allele2sklearn_metrics)
        .transpose(include_header=True, column_names=["metrics"], header_name="allele")
        .unnest("metrics")
        .sort(by="allele")
    )

    df_metrics_from_pl = (
        df.lazy().group_by("allele").agg(metrics_pl_expr).sort(by="allele").collect()
    )

    pl_testing.assert_frame_equal(
        df_metrics_from_pl, df_metrics_from_sklearn, check_column_order=False
    )


@pytest.mark.parametrize("use_sample_weight", [True, False])
def test_pl_binary_metrics_ties_probabilities(use_sample_weight: bool) -> None:
    """Ensure polars and sklearn functions have the same behavior with ties in probabilities."""
    y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    y_proba = np.array([0.4, 0.1, 0.3, 0.1, 0.1])

    sample_weight = None
    if use_sample_weight:
        sample_weight = np.random.rand(y_true.shape[0])

    expected_roc_auc = roc_auc_score(y_true, y_proba, sample_weight=sample_weight)
    expected_average_precision = average_precision_score(
        y_true, y_proba, sample_weight=sample_weight
    )
    expected_precision, expected_recall, _ = precision_recall_curve(
        y_true, y_proba, sample_weight=sample_weight
    )
    expected_pr_auc = auc(expected_recall, expected_precision)

    if use_sample_weight:
        metrics_pl = (
            pl.LazyFrame({"y_true": y_true, "y_proba": y_proba, "sample_weight": sample_weight})
            .select(
                pl.col("y_true")
                .binary_metrics.roc_auc_score(pl.col("y_proba"), pl.col("sample_weight"))
                .alias("roc_auc"),
                pl.col("y_true")
                .binary_metrics.average_precision_score(pl.col("y_proba"), pl.col("sample_weight"))
                .alias("average_precision"),
                pl.col("y_true")
                .binary_metrics.pr_auc_score(pl.col("y_proba"), pl.col("sample_weight"))
                .alias("pr_auc"),
            )
            .collect()
        )
    else:
        metrics_pl = (
            pl.LazyFrame({"y_true": y_true, "y_proba": y_proba})
            .select(
                pl.col("y_true").binary_metrics.roc_auc_score(pl.col("y_proba")).alias("roc_auc"),
                pl.col("y_true")
                .binary_metrics.average_precision_score(pl.col("y_proba"))
                .alias("average_precision"),
                pl.col("y_true").binary_metrics.pr_auc_score(pl.col("y_proba")).alias("pr_auc"),
            )
            .collect()
        )

    assert_almost_equal(metrics_pl.item(0, "roc_auc"), expected_roc_auc, decimal=9)
    assert_almost_equal(
        metrics_pl.item(0, "average_precision"), expected_average_precision, decimal=9
    )
    assert_almost_equal(metrics_pl.item(0, "pr_auc"), expected_pr_auc, decimal=9)


@pytest.mark.parametrize(
    ("sample_weight", "expected_top_k"),
    [
        (None, 0.6),
        (np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 0.6),  # same weight - no impact
        (np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 0.6),  # same weight - no impact
        (
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5]),
            0.6,
        ),  # same weight for matched peptides - no impact
        (
            np.array([1.5, 1.5, 1.0, 1.0, 1.5, 1.0]),
            0.692307692,
        ),  # higher weights for matched hits - increased top-k
        (
            np.array([0.5, 0.5, 1.0, 1.0, 0.5, 1.0]),
            0.428571428,
        ),  # lower weights for matched hits - decreased top-k
    ],
)
def test_top_k_score_same_proba(sample_weight: np.ndarray | None, expected_top_k: float) -> None:
    """Ensure top_k_score gives the expected result with edge case."""
    df = pl.DataFrame({"hit": [1, 1, 0, 0, 1, 1], "score": [0.8, 0.7, 0.7, 0.2, 0.2, 0.1]})
    if isinstance(sample_weight, np.ndarray):
        df = df.with_columns(pl.lit(sample_weight).alias("sample_weight"))

    if isinstance(sample_weight, np.ndarray):
        top_k = df.select(
            pl.col("hit").binary_metrics.top_k_score(
                pl.col("score"),
                sample_weight=pl.col("sample_weight"),
            )
        ).item()
    else:
        top_k = df.select(pl.col("hit").binary_metrics.top_k_score(pl.col("score"))).item()

    assert_almost_equal(top_k, expected_top_k, decimal=9)


@pytest.mark.parametrize("use_sample_weight", [True, False])
def test_pl_metrics_confusion_matrix(use_sample_weight: bool) -> None:
    """Ensure the TP, TN, FP and FN computed with polars are the same as sklearn."""
    num_samples = 100_000

    df = pl.DataFrame(
        {
            "y_true": (np.random.rand(num_samples) > 0.99).astype(int),
            "y_proba": np.random.rand(num_samples),
        }
    ).with_columns(
        # to compute precision, recall, ... we need the predicted labels (same as sklearn)
        (pl.col("y_proba") > 0.5).cast(pl.Int32).alias("y_pred")
    )
    if use_sample_weight:
        df = df.with_columns((pl.lit(np.random.rand(df.height))).alias("sample_weight"))

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    sample_weight = None
    if use_sample_weight:
        sample_weight = df["sample_weight"].to_numpy()

        confusion_matrix_from_pl = (
            df.lazy()
            .select(
                pl.col("y_true")
                .binary_metrics.true_negatives(
                    pl.col("y_pred"),
                    pl.col("sample_weight"),
                )
                .alias("tn"),
                pl.col("y_true")
                .binary_metrics.true_positives(
                    pl.col("y_pred"),
                    pl.col("sample_weight"),
                )
                .alias("tp"),
                pl.col("y_true")
                .binary_metrics.false_negatives(
                    pl.col("y_pred"),
                    pl.col("sample_weight"),
                )
                .alias("fn"),
                pl.col("y_true")
                .binary_metrics.false_positives(
                    pl.col("y_pred"),
                    pl.col("sample_weight"),
                )
                .alias("fp"),
            )
            .collect()
        )

    else:
        confusion_matrix_from_pl = (
            df.lazy()
            .select(
                pl.col("y_true").binary_metrics.true_negatives(pl.col("y_pred")).alias("tn"),
                pl.col("y_true").binary_metrics.true_positives(pl.col("y_pred")).alias("tp"),
                pl.col("y_true").binary_metrics.false_negatives(pl.col("y_pred")).alias("fn"),
                pl.col("y_true").binary_metrics.false_positives(pl.col("y_pred")).alias("fp"),
            )
            .collect()
        )

    (expected_tn, expected_fp, expected_fn, expected_tp) = confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight
    ).ravel()

    assert_almost_equal(confusion_matrix_from_pl.item(0, "tn"), expected_tn, decimal=9)
    assert_almost_equal(confusion_matrix_from_pl.item(0, "fp"), expected_fp, decimal=9)
    assert_almost_equal(confusion_matrix_from_pl.item(0, "fn"), expected_fn, decimal=9)
    assert_almost_equal(confusion_matrix_from_pl.item(0, "tp"), expected_tp, decimal=9)


def test_frank_score() -> None:
    """Ensure frank_score gives the expected results."""
    lf = pl.LazyFrame(
        {
            "identifier": ["id1", "id1", "id1", "id2", "id2", "id2"],
            "hit": [1, 0, 1, 1, 0, 0],
            "hit_test_dummy_best": [0.7, 0.5, 0.3, 0.6, 0.5, 0.7],
        }
    )

    df_metrics = (
        lf.group_by("identifier")
        .agg(
            pl.col("hit")
            .binary_metrics.frank_score(pl.col("hit_test_dummy_best"))
            .alias("frank_score_hit"),
        )
        .collect()
    )

    df_expected = pl.DataFrame(
        {
            "identifier": ["id1", "id2"],
            "frank_score_hit": [float("NaN"), 0.5],
        }
    )

    pl_testing.assert_frame_equal(
        df_metrics, df_expected, check_row_order=False, check_column_order=False
    )


def test_top_k_score_raise_error() -> None:
    """Ensure top_k_score raises error when sample weights are provided with specific k."""
    y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    y_proba = np.array([0.4, 0.05, 0.3, 0.05, 0.1])
    sample_weights = np.array([1.0, 0.5, 0.5, 1.0, 1.0])

    with pytest.raises(
        ValueError,
        match=(
            "For Top-K computation, if a specific value of k "
            "is requested, sample weights aren't available."
        ),
    ):
        (
            pl.DataFrame(
                {"y_true": y_true, "y_proba": y_proba, "sample_weight": sample_weights}
            ).select(
                pl.col("y_true").binary_metrics.top_k_score(
                    pl.col("y_proba"), sample_weight=pl.col("sample_weight"), k=2
                )
            )
        )
