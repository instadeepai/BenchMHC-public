# Expr.binary_metrics is added at runtime
# mypy: disable-error-code="attr-defined"
"""Module to define the command line to evaluate."""

from pathlib import Path

import numpy as np
import polars as pl

from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.logging import system

log = system.get(__name__)


def evaluate(
    dataset_path: Path,
    prediction_identifier: str,
    target_identifier: str,
    output_dir: Path,
    group_identifier: str,
    with_frank_score: bool,
) -> None:
    """Evaluate predictions against target values.

    Refer to bench_mhc.cli.evaluate_command.py::evaluate for the complete documentation.

    Args:
        dataset_path: Path to the dataset containing predictions and target values.
        prediction_identifier: Name of the column containing predictions.
        target_identifier: Name of the column containing target values.
        output_dir: Directory to save the evaluation results.
        group_identifier: Name of the column with group identifiers to compute metrics per group.
        with_frank_score: Whether to compute the Frank score. If True, only the Frank score will be
            computed. If False, all binary metrics will be computed.

    Raises:
        ValueError: If any of the following columns are not found in the dataset:
            - prediction_identifier
            - target_identifier
            - group_identifier
    """
    log.info(f"Loading dataset from '{dataset_path}'...")
    lf = pl.scan_csv(dataset_path)

    columns = lf.collect_schema().names()

    if prediction_identifier not in columns:
        raise ValueError(
            f"Column '{prediction_identifier}' not found in dataset. "
            f"Available columns: {format_iterable(columns)}"
        )

    if target_identifier not in columns:
        raise ValueError(
            f"Column '{target_identifier}' not found in dataset. "
            f"Available columns: {format_iterable(columns)}"
        )

    if group_identifier not in columns:
        raise ValueError(
            f"Column '{group_identifier}' not found in dataset. "
            f"Available columns: {format_iterable(columns)}"
        )

    if with_frank_score:
        num_epitope_with_more_than_one_hit = (
            lf.group_by(group_identifier)
            .agg(pl.col(target_identifier).sum())
            .select((pl.col(target_identifier) > 1).sum())
            .collect(engine="streaming")
            .item()
        )
        if num_epitope_with_more_than_one_hit > 0:
            raise ValueError(
                f"Some epitopes defined with the '{group_identifier}' "
                "column have more than one hit. "
                "This is not supported for the Frank Score evaluation. "
                "Make sure that each epitope has only one hit."
            )

    log.info("Dataset loaded successfully.")

    if "pctrnk" in prediction_identifier:
        log.info(
            "Detecting percentile rank prediction identifier. "
            "The ranks will be transformed to scores between 0 and 1 with the following "
            "transformation: 1 - rank/100 ."
        )
        lf = lf.with_columns((1 - pl.col(prediction_identifier) / 100).alias(prediction_identifier))

    log.info(f"Computing metrics per {group_identifier}...")

    if with_frank_score:
        metrics_query = [
            pl.col(target_identifier)
            .binary_metrics.frank_score(pl.col(prediction_identifier))
            .alias("Frank-Score"),
        ]
    else:
        metrics_query = [
            pl.col(target_identifier)
            .binary_metrics.top_k_score(pl.col(prediction_identifier))
            .alias("Top-K"),
            pl.col(target_identifier)
            .binary_metrics.pr_auc_score(pl.col(prediction_identifier))
            .alias("PR-AUC"),
            pl.col(target_identifier)
            .binary_metrics.average_precision_score(pl.col(prediction_identifier))
            .alias("AP"),
            pl.col(target_identifier)
            .binary_metrics.roc_auc_score(pl.col(prediction_identifier))
            .alias("ROC-AUC"),
        ]

    df_metrics_per_group = (
        lf.group_by(group_identifier).agg(metrics_query).sort(group_identifier).collect()
    )
    if with_frank_score:
        df_median_or_global_metrics = df_metrics_per_group.select(
            pl.lit("Median").alias(group_identifier), pl.exclude(group_identifier).median()
        )
    else:
        df_median_or_global_metrics = (
            lf.select(metrics_query)
            .with_columns(pl.lit("Global").alias(group_identifier))
            .collect()
        )
    df_mean_metrics = df_metrics_per_group.select(
        pl.lit("Mean").alias(group_identifier), pl.exclude(group_identifier).mean()
    )
    log.info("Evaluation results computed successfully.")

    # Extract and print metrics
    log.info("Summary metrics:")
    log.info("-" * 80)
    global_or_median_str = "Median" if with_frank_score else "Global"
    log.info(f"{'Metric':<10} {global_or_median_str:<10} {'Mean':<10}")
    log.info("-" * 80)

    metrics = ["Top-K", "PR-AUC", "AP", "ROC-AUC"] if not with_frank_score else ["Frank-Score"]
    for metric in metrics:
        global_or_median_value = df_median_or_global_metrics.filter(
            pl.col(group_identifier) == global_or_median_str
        )[metric].item()
        mean_value = df_mean_metrics.filter(pl.col(group_identifier) == "Mean")[metric].item()

        # Handle NaN values for prettier printing
        global_or_median_value_str = (
            f"{global_or_median_value:.4f}" if not np.isnan(global_or_median_value) else "NaN"
        )
        mean_value_str = f"{mean_value:.4f}" if not np.isnan(mean_value) else "NaN"

        log.info(f"{metric:<10} {global_or_median_value_str:<10} {mean_value_str:<10}")

    log.info("-" * 80)

    # Save output metrics file
    df_metrics = pl.concat(
        [
            df_metrics_per_group,
            df_median_or_global_metrics.select(df_metrics_per_group.columns),
            df_mean_metrics.select(df_metrics_per_group.columns),
        ],
        how="vertical",
    )

    output_file_path = (
        output_dir / f"{dataset_path.stem}_{prediction_identifier}_{target_identifier}.csv"
    )

    log.info("Saving evaluation results...")
    df_metrics.write_csv(output_file_path)
    log.info(f"Evaluation results saved successfully to {output_file_path}.")
