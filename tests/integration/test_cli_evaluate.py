"""Integration tests related to bench_mhc/cli/evaluate_command.py."""

from pathlib import Path

import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest
from click.testing import CliRunner

from bench_mhc.cli.evaluate_command import evaluate


@pytest.fixture
def evaluation_dataset(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    """Get a dataset for testing."""
    param = getattr(request, "param", "eager")
    if param == "per_epitope":
        data = {
            "epitope": ["E01", "E01", "E02", "E02", "E02"],
            "dummy_predictions_hit": [0.9, 0.1, 0.15, 0.2, 0.1],
            "dummy_predictions_pctrnk_hit": [1, 99, 50, 75, 25],
            # Expected Frank-Score: {E01: 0, E02: 0.5}
            "dummy_hit": [1, 0, 1, 0, 0],
        }
    else:
        data = {
            "allele": ["A0101", "A0101", "A0201", "A0201", "A0201"],
            "dummy_predictions_hit": [0.9, 0.1, 0.4, 0.2, 0.1],
            "dummy_predictions_pctrnk_hit": [1, 99, 55, 75, 95],
            "dummy_hit": [1, 0, 1, 0, 1],
        }

    df = pl.DataFrame(data)

    file_path = tmp_path / f"test_cli_evaluation_{param}.csv"
    df.write_csv(file_path)

    return str(file_path)


@pytest.fixture
def dataset_epitope_with_more_than_one_hit(tmp_path: Path) -> str:
    """Create a dataset with an epitope with more than one hit."""
    df = pl.DataFrame(
        {
            "epitope": ["E01", "E01", "E02", "E02", "E02"],
            "dummy_predictions_hit": [0.9, 0.1, 0.15, 0.2, 0.1],
            "dummy_hit": [1, 0, 1, 1, 0],
        }
    )

    file_path = tmp_path / "test_cli_evaluation_epitope_with_more_than_one_hit.csv"
    df.write_csv(file_path)

    return str(file_path)


@pytest.fixture
def dataset_missing_predictions(evaluation_dataset: str, tmp_path: Path) -> str:
    """Create a dataset missing the predictions column."""
    df = pl.read_csv(evaluation_dataset)
    df = df.drop("dummy_predictions_hit")

    file_path = tmp_path / "test_cli_evaluation_missing_predictions.csv"
    df.write_csv(file_path)

    return str(file_path)


@pytest.fixture
def dataset_missing_target(evaluation_dataset: str, tmp_path: Path) -> str:
    """Create a dataset missing the target column."""
    df = pl.read_csv(evaluation_dataset)
    df = df.drop("dummy_hit")

    file_path = tmp_path / "test_cli_evaluation_missing_target.csv"
    df.write_csv(file_path)

    return str(file_path)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    return tmp_path / "test_evaluate"


@pytest.mark.parametrize(
    "prediction_identifier", ["dummy_predictions_hit", "dummy_predictions_pctrnk_hit"]
)
@pytest.mark.parametrize("evaluation_dataset", ["per_allele", "per_epitope"], indirect=True)
def test_evaluate(
    prediction_identifier: str,
    evaluation_dataset: str,
    output_dir: Path,
) -> None:
    """Test the evaluate command with valid inputs."""
    group_identifier = "allele" if "per_allele" in evaluation_dataset else "epitope"
    evaluate_params = [
        "-d",
        evaluation_dataset,
        "-p_id",
        prediction_identifier,
        "-t_id",
        "dummy_hit",
        "-o",
        str(output_dir),
        "--group_identifier",
        group_identifier,
    ]

    if "per_epitope" in evaluation_dataset:
        evaluate_params.append("--with_frank_score")

    runner = CliRunner()

    result = runner.invoke(
        evaluate,
        evaluate_params,
    )

    assert result.exit_code == 0

    # Check that output file exists
    expected_output_file = (
        output_dir
        / f"test_cli_evaluation_per_{group_identifier}_{prediction_identifier}_dummy_hit.csv"
    )
    assert expected_output_file.exists()

    df_metrics = pl.read_csv(expected_output_file)

    if "per_allele" in evaluation_dataset:
        expected_metrics = {"Top-K", "PR-AUC", "AP", "ROC-AUC"}
        assert expected_metrics.issubset(df_metrics.columns)

        expected_allele_fields = {"A0101", "A0201", "Global", "Mean"}
        assert set(df_metrics["allele"]) == expected_allele_fields

        # Verify metric values are all between 0 and 1
        # It indirectly checks that all metrics are not NaN.
        assert not df_metrics.select(
            pl.sum_horizontal(~pl.exclude("allele").is_between(0, 1).all())
        ).item()

        # Assert metrics for A0101 are all 1
        assert (
            df_metrics.filter(pl.col("allele") == "A0101")
            .select(pl.sum_horizontal(pl.exclude("allele")))
            .item()
            == 4
        )

        # Use allclose for inexact global Top-K at 2/3.
        assert np.allclose(df_metrics["Top-K"], [1.0, 0.5, 2 / 3, 0.75])

    else:
        expected_metrics = {"Frank-Score"}
        assert expected_metrics.issubset(df_metrics.columns)

        expected_epitope_fields = {"E01", "E02", "Mean", "Median"}
        assert set(df_metrics["epitope"]) == expected_epitope_fields

        expected_df = pl.DataFrame(
            {"epitope": ["E01", "E02", "Mean", "Median"], "Frank-Score": [0.0, 0.5, 0.25, 0.25]}
        )

        pl_testing.assert_frame_equal(
            df_metrics,
            expected_df,
            check_row_order=False,
            check_column_order=False,
            check_dtypes=False,
        )


def test_evaluate_missing_predictions(
    dataset_missing_predictions: str,
    output_dir: Path,
) -> None:
    """Test the evaluate command with missing predictions column."""
    runner = CliRunner()

    match_msg = "Column 'dummy_predictions_hit' not found"
    with pytest.raises(ValueError, match=match_msg):
        runner.invoke(
            evaluate,
            [
                "-d",
                dataset_missing_predictions,
                "-p_id",
                "dummy_predictions_hit",
                "-t_id",
                "dummy_hit",
                "-o",
                str(output_dir),
            ],
            catch_exceptions=False,
        )


def test_evaluate_missing_target(
    dataset_missing_target: str,
    output_dir: Path,
) -> None:
    """Test the evaluate command with missing target column."""
    runner = CliRunner()

    match_msg = "Column 'dummy_hit' not found"
    with pytest.raises(ValueError, match=match_msg):
        runner.invoke(
            evaluate,
            [
                "-d",
                dataset_missing_target,
                "-p_id",
                "dummy_predictions_hit",
                "-t_id",
                "dummy_hit",
                "-o",
                str(output_dir),
            ],
            catch_exceptions=False,
        )


def test_evaluate_missing_group(
    evaluation_dataset: str,
    output_dir: Path,
) -> None:
    """Test the evaluate command with missing group column."""
    runner = CliRunner()

    match_msg = "Column 'unknown_group_column' not found"
    with pytest.raises(ValueError, match=match_msg):
        runner.invoke(
            evaluate,
            [
                "-d",
                evaluation_dataset,
                "-p_id",
                "dummy_predictions_hit",
                "-t_id",
                "dummy_hit",
                "-o",
                str(output_dir),
                "-g_id",
                "unknown_group_column",
            ],
            catch_exceptions=False,
        )


def test_evaluate_epitope_with_more_than_one_hit(
    dataset_epitope_with_more_than_one_hit: str,
) -> None:
    """Test the evaluate command with epitope with more than one hit."""
    runner = CliRunner()
    evaluate_params = [
        "-d",
        dataset_epitope_with_more_than_one_hit,
        "-p_id",
        "dummy_predictions_hit",
        "-t_id",
        "dummy_hit",
        "-o",
        "dummy_output_dir",
        "--group_identifier",
        "epitope",
        "--with_frank_score",
    ]

    match_msg = (
        "Some epitopes defined with the 'epitope' column have more than one hit. "
        "This is not supported for the Frank Score evaluation. "
        "Make sure that each epitope has only one hit."
    )
    with pytest.raises(ValueError, match=match_msg):
        runner.invoke(
            evaluate,
            evaluate_params,
            catch_exceptions=False,
        )
