"""Configuration file for shared fixtures used in tests."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from bench_mhc.constants import NATURAL_AAS
from bench_mhc.constants import SEPARATOR
from bench_mhc.dataset.main import Dataset
from bench_mhc.utils.io import save_json
from bench_mhc.utils.io import save_yml
from bench_mhc.utils.mode import Mode
from bench_mhc.variables.variables import Outputs
from bench_mhc.variables.variables import Variables


@pytest.fixture
def dataframe(request: pytest.FixtureRequest) -> pl.DataFrame | pl.LazyFrame:
    """Fixture used to define a generic DataFrame to use in most tests."""
    df = pl.DataFrame(
        {
            "aa_seq": ["TEST", "PEP", "TT", "EE", None, "TT"],
            "aa_seq_to_map": ["A", "B", "C", "D", "E", "F"],
            "aa_seq_to_map_unknown_key": ["A", "B", "C", "D", "E", "unknown"],
            "binary": [0, 1, 0, 1, 0, 1],
            "binary_nan": [0, 1, None, 1, 0, None],
            "binary_nan_already_replaced": [0, 1, -1, 1, 0, None],
            "binary_float": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "binary_float_nan": [0.0, 1.0, None, 1.0, 0.0, np.nan],
            "binary_str": ["0", "1", "0", "1", "0", "1"],
            "binary_str_nan": ["0", "1", "NaN", "1", "0", "NaN"],
            "binary_non_binary": [0, 1, None, 2, -1, None],
            "n_flank": [
                "FLANKINGA",
                "INGC",
                "FLANGE",
                "FLANKI",
                "FLANKINGI",
                "FLANKINGK",
            ],
            "c_flank": [
                "FLANKINGB",
                "KINGD",
                "FLANKINGF",
                "INGH",
                "FLANKINGJ",
                "FLANKINGL",
            ],
            "numeric": [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
            "numeric_nan": [0.1, 0.2, None, -0.1, -0.2, np.nan],
            "numeric_str": ["0.1", "0.2", "0.3", "-0.1", "-0.2", "-0.3"],
            "numeric_str_nan": ["0.1", "0.2", "NaN", "-0.1", "-0.2", "NaN"],
            "nn_align": [
                "SIXMER",
                "EIGHTMER",
                "NINEMEEER",
                "ITSATENMER",
                "ITSTWELVEMER",
                None,
            ],
            "nn_align_to_map": ["A", "B", "C", "D", "E", "F"],
            "peptide": ["PEPTIDEA", "PEPTIDEC", "PEPTIDEE", "PEPTIDEG", "PEPTIDEI", "PEPTIDEK"],
            "peptide_to_map": [
                "PEPTIDEB",
                "PEPTIDED",
                "PEPTIDEF",
                "PEPTIDEH",
                "PEPTIDEJ",
                "PEPTIDEL",
            ],
            "random": ["R", "A", "N", "D", "O", "M"],
            "allele": [
                "HLA-A*02:01",
                "HLA-DRB1*02:01",
                "DPA10301-DPB10305",
                "DRA10101=DRB10201",
                "unparsable",
                "Mamu_A2:0511",
            ],
        }
    )

    param = getattr(request, "param", "eager")
    if param == "lazy":
        return df.lazy()

    return df


@pytest.fixture
def alleles() -> list[str]:
    """Define the alleles from the MHC1 dataframe."""
    return [
        "Mamu__B04610",
        "Mamu__B04610",
        "BoLA__106701",
        "BoLA__106701",
        "HLA__B5809",
        "HLA__B5809",
    ]


@pytest.fixture
def mhc1_dataframe(alleles: list[str]) -> pl.DataFrame:
    """Fixture used to define an MHC1 dataframe."""
    df = pl.DataFrame(
        {
            "peptide": ["TESTPEP", "PEPTEST", "TT", "EE", None, "TT"],
            "allele": alleles,
            "hit": [0, 1, 0, 1, 0, None],
            "binding_affinity": [0.2, 0.1, None, 0.8, 0, 0.9],
        }
    )

    return df


@pytest.fixture
def ma_dataframe(request: pytest.FixtureRequest) -> pl.LazyFrame | pl.DataFrame:
    """Fixture used to define a multi-allelic (MA) dataframe.

    It contains MHC1 & MHC2 alleles.
    """
    df = pl.DataFrame(
        data={
            "allele": [
                "HLA__A0201",
                "HLA__B0702",
                "HLA__DQA10101=DQB10201",
                "HLA__DRB11610",
                "HLA__C0102",
                "HLA__DPA10103=DPB10101",
                "HLA__DRB10101",
                "HLA__DQA10101=DQB10201",
                "HLA__A0201",
                "HLA__A0103",
            ],
            "peptide": [
                "AAAA",
                "AAAA",
                "BBBB",
                "BBBB",
                "AAAA",
                "CCCC",
                "CCCC",
                "CCCC",
                "AAAA",
                "AAAA",
            ],
            "sample_alias": [
                "exp1",
                "exp1",
                "exp1",
                "exp1",
                "exp1",
                "exp2",
                "exp2",
                "exp2",
                "exp2",
                "exp2",
            ],
            "random": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "predictions": [0.6, 0.7, 0.8, 0.9, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1],
            "hit": [1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            "binding_affinity": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        }
    )
    df = df.with_columns(
        pl.concat_str("peptide", "sample_alias", separator=SEPARATOR).alias("MA_bag_identifier"),
    )

    param = getattr(request, "param", "eager")
    if param == "lazy":
        return df.lazy()

    return df


@pytest.fixture
def expected_nn_align_processed_mhc1_df(dataframe: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Get the expected NNAlign-processed data for MHC1."""
    expected_data = pl.DataFrame(
        data={
            "peptide": dataframe.lazy().collect()["nn_align"].to_list(),
            "random": dataframe.lazy().collect()["random"].to_list(),
            "allele": dataframe.lazy().collect()["allele"].to_list(),
            "nn_align_num_possible_cores": [7, 9, 1, 10, 10, 1],
            "nn_align_core_0": [
                "---SIXMER",
                "-EIGHTMER",
                "NINEMEEER",
                "TSATENMER",
                "TWELVEMER",
                "---------",
            ],
            "nn_align_core_0_flank_left_len": [0, 0, 0, 1, 3, 0],
            "nn_align_core_0_flank_right_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_1": [
                "S---IXMER",
                "E-IGHTMER",
                "---------",
                "ISATENMER",
                "IWELVEMER",
                "---------",
            ],
            "nn_align_core_1_flank_left_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_1_flank_right_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_2": [
                "SI---XMER",
                "EI-GHTMER",
                "---------",
                "ITATENMER",
                "ITELVEMER",
                "---------",
            ],
            "nn_align_core_2_flank_left_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_2_flank_right_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_3": [
                "SIX---MER",
                "EIG-HTMER",
                "---------",
                "ITSTENMER",
                "ITSLVEMER",
                "---------",
            ],
            "nn_align_core_3_flank_left_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_3_flank_right_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_4": [
                "SIXM---ER",
                "EIGH-TMER",
                "---------",
                "ITSAENMER",
                "ITSTVEMER",
                "---------",
            ],
            "nn_align_core_4_flank_left_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_4_flank_right_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_5": [
                "SIXME---R",
                "EIGHT-MER",
                "---------",
                "ITSATNMER",
                "ITSTWEMER",
                "---------",
            ],
            "nn_align_core_5_flank_left_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_5_flank_right_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_6": [
                "SIXMER---",
                "EIGHTM-ER",
                "---------",
                "ITSATEMER",
                "ITSTWEMER",
                "---------",
            ],
            "nn_align_core_6_flank_left_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_6_flank_right_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_7": [
                "---------",
                "EIGHTME-R",
                "---------",
                "ITSATENER",
                "ITSTWELER",
                "---------",
            ],
            "nn_align_core_7_flank_left_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_7_flank_right_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_8": [
                "---------",
                "EIGHTMER-",
                "---------",
                "ITSATENMR",
                "ITSTWELVR",
                "---------",
            ],
            "nn_align_core_8_flank_left_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_8_flank_right_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_9": [
                "---------",
                "---------",
                "---------",
                "ITSATENME",
                "ITSTWELVE",
                "---------",
            ],
            "nn_align_core_9_flank_left_len": [0, 0, 0, 0, 0, 0],
            "nn_align_core_9_flank_right_len": [0, 0, 0, 1, 3, 0],
            "nn_align_insertion_len": [3, 1, 0, 0, 0, 9],
            "nn_align_deletion_len": [0, 0, 0, 1, 3, 0],
            "nn_align_is_8mer_or_less": [1, 1, 0, 0, 0, 1],
            "nn_align_is_9mer": [0, 0, 1, 0, 0, 0],
            "nn_align_is_10mer": [0, 0, 0, 1, 0, 0],
            "nn_align_is_11mer_or_more": [0, 0, 0, 0, 1, 0],
        }
    )

    return expected_data


@pytest.fixture
def aa_seq_mapping_file_path(tmp_path: Path) -> str:
    """Define the file path for the mapping to use in tests relative to AASeqVariable."""
    mapping = {
        "A": "TEST",
        "B": "PEP",
        "C": "TT",
        "D": "EE",
        "E": None,
        "F": "TT",
    }
    mapping_file_path = tmp_path / "aa_seq_mapping.json"
    save_json(mapping, mapping_file_path)

    return str(mapping_file_path)


@pytest.fixture
def variables_configuration(aa_seq_mapping_file_path: str) -> dict[str, dict[str, Any]]:
    """Get variables configuration for the tests."""
    return {
        "inputs": {
            "binary": {
                "class_name": "BinaryVariable",
                "default_value": -1,
                "column": "binary_nan",
            },
            "numeric": {
                "class_name": "NumericVariable",
                "default_value": -1,
            },
            "nn_align": {
                "class_name": "NNAlignVariable",
                "unk_token": "X",
                "pad_token": "_",
                "aas": ["T", "E", "S", "T"],
            },
            "aa_seq_to_map": {
                "class_name": "AASeqVariable",
                "unk_token": "X",
                "pad_token": "_",
                "aas": ["T", "E", "S", "T"],
                "max_len": 10,
                "padding_strategy": "start",
                "mapping_file_path": Path(aa_seq_mapping_file_path),
            },
        },
        "outputs": {
            "binary": {
                "class_name": "BinaryOutput",
                "default_value": -1,
                "loss": {"class_name": "BCELoss", "reduction": "sum"},
            },
            "numeric": {
                "class_name": "NumericOutput",
                "default_value": -1,
                "metrics": [
                    {"class_name": "MeanSquaredError", "squared": False},
                    {"class_name": "MeanAbsoluteError", "name": "mean_absolute_error"},
                ],
            },
        },
    }


@pytest.fixture
def mhc1_nn_align_configuration() -> dict[str, dict[str, Any]]:
    """Get MHC1NNAlignModel configuration for the tests."""
    return {
        "variables": {
            "inputs": {
                "peptide": {
                    "class_name": "NNAlignVariable",
                    "unk_token": "X",
                    "aas": list(NATURAL_AAS),
                },
                "allele": {
                    "class_name": "AASeqVariable",
                    "unk_token": "X",
                    "aas": list(NATURAL_AAS),
                    "mapping_file_path": "data/mappings/allele2netmhc_pseudo_seq.json",
                },
            },
            "outputs": {
                "hit": {
                    "class_name": "BinaryOutput",
                    "default_value": -1,
                },
                "binding_affinity": {
                    "class_name": "NumericOutput",
                    "default_value": -1,
                },
            },
        },
        "model": {"class_name": "MHC1NNAlignModel", "hidden_dim": None},
        "training": {
            "logger": {"class_name": "CSVLogger"},
            "epochs": 2,
            "batch_size": 4,
            "optimizer": {
                "class_name": "SGD",
                "lr": 0.05,
            },
            "num_workers": 2,
            "prefetch_factor": 2,
        },
    }


@pytest.fixture
def mhc1_nn_align_inputs(mhc1_nn_align_configuration: dict[str, dict[str, Any]]) -> Variables:
    """Input variables for the tests with MHC1NNAlignModel."""
    return Variables.from_dict(mhc1_nn_align_configuration["variables"]["inputs"])


@pytest.fixture
def mhc1_nn_align_outputs(mhc1_nn_align_configuration: dict[str, dict[str, Any]]) -> Outputs:
    """Output variables for the tests with MHC1NNAlignModel."""
    return Outputs.from_dict(mhc1_nn_align_configuration["variables"]["outputs"])


@pytest.fixture
def mhc1_nn_align_dataset(
    mhc1_nn_align_inputs: Variables, mhc1_nn_align_outputs: Outputs, mhc1_dataframe: pl.DataFrame
) -> Dataset:
    """Fixture used to define dataset to use to test MHC1NNAlignModel.

    Input and output variables are fitted and used to transform the dataframe.
    """
    variables = mhc1_nn_align_inputs + mhc1_nn_align_outputs
    variables.fit(mhc1_dataframe)

    transformed_df = variables.transform(mhc1_dataframe)

    dataset = Dataset(
        df=transformed_df, mode=Mode.VAL, inputs=mhc1_nn_align_inputs, outputs=mhc1_nn_align_outputs
    )

    return dataset


@pytest.fixture
def model_directory(tmp_path: Path) -> Path:
    """Get the model directory for tests."""
    model_directory_path = tmp_path / "models"
    model_directory_path.mkdir()

    return model_directory_path


@pytest.fixture
def configuration_file_path(tmp_path: Path) -> str:
    """Get model configuration file path for the tests."""
    file_path = tmp_path / "configuration.yaml"

    configuration = {
        "variables": {
            "inputs": {
                "peptide": {
                    "class_name": "NNAlignVariable",
                    "unk_token": "X",
                    "aas": list(NATURAL_AAS),
                },
                "allele": {
                    "class_name": "AASeqVariable",
                    "unk_token": "X",
                    "aas": list(NATURAL_AAS),
                    "mapping_file_path": "data/mappings/allele2netmhc_pseudo_seq.json",
                },
            },
            "outputs": {
                "hit": {
                    "class_name": "BinaryOutput",
                    "default_value": -1,
                },
                "binding_affinity": {
                    "class_name": "NumericOutput",
                    "default_value": -1,
                },
            },
        },
        "model": {"class_name": "MHC1NNAlignModel", "hidden_dim": None},
        "training": {
            "logger": {"class_name": "CSVLogger"},
            "epochs": 4,
            "batch_size": 4,
            "optimizer": {
                "class_name": "SGD",
                "lr": 0.05,
            },
            "num_workers": 0,
            "prefetch_factor": None,
            "callbacks": [
                {"class_name": "EarlyStopping", "patience": 1, "monitor": "val/hit/loss"}
            ],
            "gpus": None,
        },
    }

    save_yml(configuration, file_path)

    return str(file_path)


@pytest.fixture
def cache_directory(tmp_path: Path) -> Path:
    """Get the cache directory for tests."""
    cache_directory_path = tmp_path / ".cache"
    cache_directory_path.mkdir()

    return cache_directory_path


@pytest.fixture
def training_path(mhc1_dataframe: pl.DataFrame, tmp_path: Path) -> str:
    """Get the training file path."""
    file_path = tmp_path / "training.csv"

    mhc1_dataframe.write_csv(file_path)

    return str(file_path)


@pytest.fixture
def reference_path(tmp_path: Path) -> str:
    """Fixture to create the reference dataset."""
    reference_path = tmp_path / "reference_peptides.csv"
    df = pl.DataFrame(
        {
            "peptide": ["ABAB", "BABA", "BAAB", "ABBA", "QERHREA", "WXWNES"],
        }
    )
    df.write_csv(reference_path)

    return str(reference_path)


@pytest.fixture
def allele_mapping_path(tmp_path: Path, alleles: list[str]) -> Path:
    """Create a dummy allele mapping for calibration."""
    pseudo_sequences = [
        "XXYFLRSGGQTGHVLVFPYTYYDYRTETVYETPT",
        "XXYFLRSGGQTGHVLVFPYTYYDYRTETVYETPT",
        "YESYYREKAGQWFVSNLYLQSLFYTWSAYAYEWY",
    ]
    mapping = dict(zip(sorted(set(alleles)), pseudo_sequences, strict=True))
    allele_mapping_path = tmp_path / "allele_mapping.json"
    save_json(mapping, allele_mapping_path)

    return allele_mapping_path
