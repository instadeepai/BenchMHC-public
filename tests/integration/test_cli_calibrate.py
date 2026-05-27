"""Integration tests for the command lines in bench_mhc/cli/calibrate_command.py."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from bench_mhc.cli.calibrate_command import calibrate
from bench_mhc.cli.train_command import train
from bench_mhc.utils.io import save_txt
from bench_mhc.utils.model import get_directory_name


def _get_alleles_from_arg(alleles_arg: str | None, alleles: list[str]) -> list[str]:
    """Get list of alleles from the alleles_arg parameter."""
    if alleles_arg is None:
        return sorted(set(alleles))[:3]

    if "," in alleles_arg:
        return alleles_arg.split(",")

    with open(alleles_arg) as f:
        return [line.strip() for line in f]


def _prepare_prebuilt_reference(
    reference_path: str,
    alleles_arg: str | None,
    alleles: list[str],
    tmp_path: Path,
) -> str:
    """Create reference file with peptides and alleles for a prebuilt reference set."""
    df_ref = pl.scan_csv(reference_path).collect(engine="streaming")
    alleles_to_use = _get_alleles_from_arg(alleles_arg, alleles)

    df_alleles = pl.DataFrame({"allele": alleles_to_use})
    df_prebuilt_ref = df_ref.join(df_alleles, how="cross")

    prebuilt_reference_path = tmp_path / "prebuilt_reference.csv"
    df_prebuilt_ref.write_csv(prebuilt_reference_path)

    return str(prebuilt_reference_path)


def _get_alleles_to_check(
    peptides_only: bool,
    alleles_arg: str | None,
    alleles: list[str],
) -> list[str]:
    """Get the list of alleles to check in assertions."""
    if peptides_only:
        if alleles_arg is not None:
            return ["HLA__B5809", "BoLA__106701"]
        return ["Mamu__B04610", "BoLA__106701", "HLA__B5809"]

    return _get_alleles_from_arg(alleles_arg, alleles)


def _verify_peptides_only_bin_edges(
    output_directory: Path,
    ensemble: bool,
    alleles_arg: str | None,
) -> None:
    """Verify bin edges for peptides_only reference sets."""
    expected_identical_allele = ["HLA__B5809", "BoLA__106701"]
    head_types = ["hit", "binding_affinity"] if ensemble else ["hit"]

    for head_type in head_types:
        identical_expected_bin_edges = [
            np.load(output_directory / f"{allele}__{head_type}.npz")["bin_edges"]
            for allele in expected_identical_allele
        ]
        assert not np.all(np.diff(identical_expected_bin_edges, axis=0))

    if alleles_arg is None:
        expected_different_allele = ["Mamu__B04610", "BoLA__106701"]
        for head_type in head_types:
            different_expected_bin_edges = [
                np.load(output_directory / f"{allele}__{head_type}.npz")["bin_edges"]
                for allele in expected_different_allele
            ]
            assert np.any(np.diff(different_expected_bin_edges, axis=0))


def _verify_prebuilt_bin_edges(
    output_directory: Path,
    ensemble: bool,
    allele_to_check: list[str],
) -> None:
    """Verify bin edges for prebuilt reference sets."""
    if len(allele_to_check) >= 2:
        head_types = ["hit", "binding_affinity"] if ensemble else ["hit"]
        for head_type in head_types:
            bin_edges_list = [
                np.load(output_directory / f"{allele}__{head_type}.npz")["bin_edges"]
                for allele in allele_to_check[:2]
            ]
            assert np.any(np.diff(bin_edges_list, axis=0))


@pytest.fixture
def alleles_arg(
    request: pytest.FixtureRequest, tmp_path: Path, alleles: list[str]
) -> list[str] | str | None:
    """Return a subset of alleles to calibrate."""
    param = getattr(request, "param", None)
    alleles = sorted(set(alleles))[:2]

    if param == "comma_separated_alleles":
        return ",".join(alleles)

    elif param == "alleles_file":
        allele_fp = tmp_path / "alleles.txt"
        save_txt(alleles, allele_fp)

        return str(allele_fp)
    return None


@pytest.mark.parametrize(
    "alleles_arg", [None, "comma_separated_alleles", "alleles_file"], indirect=True
)
@pytest.mark.parametrize("ensemble", [True, False])
@pytest.mark.parametrize("peptides_only", [True, False])
@pytest.mark.parametrize("apply_sigmoid", [True, False])
def test_cli_calibrate(
    alleles_arg: str | None,
    ensemble: bool,
    peptides_only: bool,
    apply_sigmoid: bool,
    model_directory: Path,
    cache_directory: Path,
    configuration_file_path: str,
    configuration_file_path_hit_only: str,
    configuration_file_path_with_logits_loss: str,
    configuration_file_path_hit_only_with_logits_loss: str,
    reference_path: str,
    tmp_path: Path,
    allele_mapping_path: Path,
    training_path: str,
    alleles: list[str],
) -> None:
    """Test the calibrate command.

    This test:
    1. Trains an ensemble of 2 models (one with hit only, one with hit and binding_affinity)
    2. Runs calibration on the ensemble
    3. Verifies that percentile rank files are produced for each allele and output type
    4. Tests with and without BCEWithLogitsLoss (apply_sigmoid parameter)
    """
    experiment_name = "test_calibrate"
    training_parameters = [
        "--experiment_name",
        experiment_name,
        "--training_path",
        training_path,
        "--validation_path",
        training_path,
        "--random_seed",
        str(42),
    ]

    model_names = [
        get_directory_name(experiment_name),
        get_directory_name(experiment_name),
    ]

    if apply_sigmoid:
        cfg_paths = [
            configuration_file_path_hit_only_with_logits_loss,
            configuration_file_path_with_logits_loss,
        ]
    else:
        cfg_paths = [configuration_file_path_hit_only, configuration_file_path]

    # Train the models
    runner = CliRunner()
    model_paths = []

    with (
        patch("bench_mhc.cli.train.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.cli.train.CACHE_DIRECTORY", cache_directory),
        patch("bench_mhc.cli.train.get_directory_name", side_effect=model_names),
    ):
        for model_name, cfg_path in zip(model_names, cfg_paths, strict=True):
            training_parameters_ = training_parameters + [
                "--configuration_file_path",
                cfg_path,
            ]
            result = runner.invoke(train, training_parameters_)
            assert result.exit_code == 0
            model_paths.append(model_directory / model_name)

            if not ensemble:
                break

        # Create model file
        if ensemble:
            model_path = tmp_path / "ensemble.txt"
            save_txt([str(model_path) for model_path in model_paths], model_path)
        else:
            model_path = model_paths[0]

        # Run calibration
        output_directory = tmp_path / "calibration_output"

        # Prepare reference path based on peptides_only flag
        if not peptides_only:
            ref_path_to_use = _prepare_prebuilt_reference(
                reference_path, alleles_arg, alleles, tmp_path
            )
        else:
            ref_path_to_use = reference_path

        calibrate_parameters = [
            "--model_path",
            str(model_path),
            "--output_directory",
            str(output_directory),
            "--reference_path",
            ref_path_to_use,
            "--batch_size",
            "2",
            "--num_workers",
            "2",
        ]
        if peptides_only:
            calibrate_parameters.append("--peptides_only")
        if alleles_arg is not None and peptides_only:
            calibrate_parameters.extend(["--alleles", alleles_arg])

        with (
            patch("bench_mhc.cli.calibrate.ALLELE_MAPPING_PATH", allele_mapping_path),
            patch("bench_mhc.cli.calibrate.NUM_STEPS_PCTRNK", 3),
        ):
            result = runner.invoke(calibrate, calibrate_parameters)

        assert result.exit_code == 0

        # Verify that percentile rank files were created
        pctrnk_files = list(output_directory.glob("*.npz"))
        allele_to_check = _get_alleles_to_check(peptides_only, alleles_arg, alleles)

        expected_pctrnk_files = {f"{allele}__hit.npz" for allele in allele_to_check}
        if ensemble:
            for allele in allele_to_check:
                expected_pctrnk_files.add(f"{allele}__binding_affinity.npz")

        assert {pctrnk_file.name for pctrnk_file in pctrnk_files} == expected_pctrnk_files

        # Verify the structure of the percentile rank files
        for pctrnk_file in pctrnk_files:
            data = np.load(pctrnk_file)
            assert "steps" in data
            assert "bin_edges" in data
            assert len(data["steps"]) == len(data["bin_edges"])
            assert data["bin_edges"][0] == 0
            assert data["bin_edges"][-1] == 1
            assert np.all(data["bin_edges"] >= 0)
            assert np.all(data["bin_edges"] <= 1)

        # Verify bin edges based on peptides_only flag
        if peptides_only:
            _verify_peptides_only_bin_edges(output_directory, ensemble, alleles_arg)
        else:
            _verify_prebuilt_bin_edges(output_directory, ensemble, allele_to_check)
