"""Integration tests related to bench_mhc/cli/generate_performance_plot_command.py."""

from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from bench_mhc.cli.generate_performance_plot_command import generate_performance_plot
from bench_mhc.utils.io import save_yml


@pytest.fixture
def configuration_file(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """Create a sample configuration file for testing.

    The fixture creates a CSV file with test metrics data and a YAML configuration
    file that references it. The metrics data varies based on the parametrized value:
    - 'ms_ligands': Creates allele-based metrics.
    - 'cd8_epitopes': Creates epitope-based metrics.

    Args:
        request: Pytest fixture request object containing the parametrized value.
        tmp_path: Temporary directory path fixture.

    Returns:
        Path: Path to the generated configuration file.
    """
    if request.param == "ms_ligands":
        df = pl.DataFrame(
            {
                "allele": ["HLA-A*01:01", "HLA-A*02:01", "HLA-B*07:02", "HLA-B*08:01"],
                "Top-K": [0.8, 0.7, 0.9, 0.85],
            }
        )
    else:
        df = pl.DataFrame(
            {
                "epitope": ["Epitope1", "Epitope2", "Epitope3", "Epitope4"],
                "Frank Score": [0.8, 0.7, 0.9, 0.85],
            }
        )

    metrics_path = tmp_path / "metrics.csv"
    df.write_csv(metrics_path)

    config_path = tmp_path / f"config_{request.param}.yml"
    config = {
        "base_model": {"metrics_path": str(metrics_path)},
        "bright_model": {"metrics_path": str(metrics_path), "bright_version_of": "base_model"},
    }

    save_yml(config, config_path)

    return config_path


@pytest.mark.parametrize("configuration_file", ["ms_ligands", "cd8_epitopes"], indirect=True)
def test_generate_performance_plot(
    configuration_file: Path,
    tmp_path: Path,
) -> None:
    """Test that the generate_performance_plot command creates the expected plot files."""
    output_dir = tmp_path / "output"

    params = [
        "--configuration_file_path",
        str(configuration_file),
        "--output_directory",
        str(output_dir),
    ]

    if "cd8_epitopes" in configuration_file.name:
        params.append("--cd8_epitopes")
    elif "ms_ligands" in configuration_file.name:
        params.append("--ms_ligands")

    runner = CliRunner()
    result = runner.invoke(
        generate_performance_plot,
        params,
    )
    assert result.exit_code == 0

    expected_file_names = (
        {"ms_ligands_per_allele_bar_plot.png", "ms_ligands_per_allele_box_plot.png"}
        if "ms_ligands" in configuration_file.name
        else {"cd8_epitopes_per_epitope_box_plot.png"}
    )

    for file_name in expected_file_names:
        assert (output_dir / file_name).exists()


@pytest.mark.parametrize("configuration_file", ["ms_ligands"], indirect=True)
def test_generate_performance_plot_wrong_flag_combination(
    configuration_file: Path,
    tmp_path: Path,
) -> None:
    """Test that the generate_performance_plot command raises the correct errors.

    It checks that the command raises an error if the wrong combination of flag is provided.
    It also checks that the command raises an error if no flag is provided.
    """
    runner = CliRunner()
    with pytest.raises(
        ValueError,
        match="You should provide either the --cd8_epitopes or --ms_ligands flag, but not both.",
    ):
        runner.invoke(
            generate_performance_plot,
            [
                "--configuration_file_path",
                str(configuration_file),
                "--output_directory",
                str(tmp_path),
                "--cd8_epitopes",
                "--ms_ligands",
            ],
            catch_exceptions=False,
        )

    with pytest.raises(
        ValueError,
        match="You should provide exactly one of the --cd8_epitopes or --ms_ligands flag.",
    ):
        runner.invoke(
            generate_performance_plot,
            [
                "--configuration_file_path",
                str(configuration_file),
                "--output_directory",
                str(tmp_path),
            ],
            catch_exceptions=False,
        )
