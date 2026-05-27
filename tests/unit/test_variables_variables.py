"""Unit tests related to bench_mhc/variables/variables.py."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.utils.errors import NotFittedError
from bench_mhc.utils.format import format_iterable
from bench_mhc.variables import NNAlignVariable
from bench_mhc.variables.variables import Outputs
from bench_mhc.variables.variables import Variables


@pytest.fixture
def expected_variables_configuration(
    aa_seq_mapping_file_path: str,
) -> dict[str, dict[str, Any]]:
    """Get the expected variables configuration for the tests."""
    return {
        "inputs": {
            "binary": {
                "name": "binary",
                "column": "binary_nan",
                "is_fitted": True,
                "default_value": -1,
                "class_name": "BinaryVariable",
            },
            "numeric": {
                "name": "numeric",
                "column": "numeric",
                "is_fitted": True,
                "default_value": -1,
                "class_name": "NumericVariable",
            },
            "nn_align": {
                "name": "nn_align",
                "column": "nn_align",
                "is_fitted": True,
                "_aas": ["_", "X", "E", "S", "T"],
                "pad_token": "_",
                "unk_token": "X",
                "_max_len": 9,
                "padding_strategy": "end",
                "mapping_file_path": None,
                "aa2idx": {"_": 0, "X": 1, "E": 2, "S": 3, "T": 4},
                "class_name": "NNAlignVariable",
            },
            "aa_seq_to_map": {
                "name": "aa_seq_to_map",
                "column": "aa_seq_to_map",
                "is_fitted": True,
                "_aas": ["_", "X", "E", "S", "T"],
                "pad_token": "_",
                "unk_token": "X",
                "_max_len": 10,
                "padding_strategy": "start",
                "mapping_file_path": aa_seq_mapping_file_path,
                "aa2idx": {"_": 0, "X": 1, "E": 2, "S": 3, "T": 4},
                "class_name": "AASeqVariable",
            },
        },
        "outputs": {
            "binary": {
                "name": "binary",
                "column": "binary",
                "is_fitted": True,
                "default_value": -1,
                "loss_config": {"class_name": "BCELoss", "reduction": "sum"},
                "metrics_config": None,
                "class_name": "BinaryOutput",
            },
            "numeric": {
                "name": "numeric",
                "column": "numeric",
                "is_fitted": True,
                "default_value": -1,
                "loss_config": None,
                "metrics_config": [
                    {"class_name": "MeanSquaredError", "squared": False},
                    {"class_name": "MeanAbsoluteError", "name": "mean_absolute_error"},
                ],
                "class_name": "NumericOutput",
            },
        },
    }


def test_init_add_variables(
    variables_configuration: dict[str, dict[str, Any]],
) -> None:
    """Test initialization and addition of variables in Variables."""
    variables_configuration = variables_configuration["inputs"]
    numeric_variable_configuration = variables_configuration.pop("numeric")

    variables = Variables.from_dict(variables_configuration)

    new_variables = Variables(list(variables))

    assert len(new_variables) == 3
    assert {"binary", "nn_align", "aa_seq_to_map"}.issubset(new_variables.names)

    new_variables += Variables.from_dict({"numeric": numeric_variable_configuration})

    assert len(new_variables) == 4
    assert "numeric" in new_variables.names

    match_msg = "must be Variables, not <class 'int'>"
    with pytest.raises(TypeError, match=match_msg):
        new_variables += 1  # type: ignore


@pytest.mark.parametrize("dataframe", ["lazy", "eager"], indirect=True)
def test_variables(
    variables_configuration: dict[str, dict[str, Any]],
    expected_variables_configuration: dict[str, dict[str, Any]],
    dataframe: LazyOrDataFrame,
) -> None:
    """Test loading the variables from configuration, fitting, serializing and transforming."""
    variables = Variables.from_dict(variables_configuration["inputs"])

    assert len(variables) == 4
    assert variables.names == set(variables_configuration["inputs"])

    expected_features = {"binary", "numeric", "aa_seq_to_map"}.union(
        {f"nn_align_{suffix}" for suffix in NNAlignVariable.suffixes}
    )
    assert variables.features == expected_features

    assert not variables.is_fitted
    with pytest.raises(NotFittedError):
        variables.transform(dataframe)

    variables.fit(dataframe)
    assert variables.is_fitted

    # Check serialization works properly
    assert (
        variables.from_dict(variables.to_dict()).to_dict()
        == expected_variables_configuration["inputs"]
    )

    for variable_name, expected_variable_configuration in expected_variables_configuration[
        "inputs"
    ].items():
        expected_variable_configuration.pop("class_name")
        assert variables[variable_name].is_fitted
        assert variables[variable_name].to_dict() == expected_variable_configuration

        # Mock each variable transform to check Variables.transform
        variables[variable_name].transform = MagicMock(wraps=variables[variable_name].transform)

    unknown_variable_name = "unknown"
    match_msg = (
        f"Variable with name '{unknown_variable_name}' not found. The available variables are the "
        f"following: {format_iterable(variable.name for variable in variables)}."
    )
    with pytest.raises(KeyError, match=match_msg):
        _ = variables[unknown_variable_name]

    _ = variables.transform(dataframe)

    for variable in variables:
        variable.transform.assert_called_once()


@pytest.mark.parametrize("dataframe", ["lazy", "eager"], indirect=True)
def test_outputs(
    variables_configuration: dict[str, dict[str, Any]],
    expected_variables_configuration: dict[str, dict[str, Any]],
    dataframe: LazyOrDataFrame,
) -> None:
    """Test loading the outputs from configuration, fitting, serializing and transforming."""
    outputs = Outputs.from_dict(variables_configuration["outputs"])

    assert outputs.names == set(variables_configuration["outputs"])

    expected_features = {"binary", "numeric"}
    assert outputs.features == expected_features

    assert not outputs.is_fitted
    with pytest.raises(NotFittedError):
        outputs.transform(dataframe)

    outputs.fit(dataframe)
    assert outputs.is_fitted

    # Check serialization works properly
    assert (
        outputs.from_dict(outputs.to_dict()).to_dict()
        == expected_variables_configuration["outputs"]
    )

    for output in outputs:
        assert output.is_fitted

        # Mock each variable transform to check Variables.transform
        output.transform = MagicMock(wraps=output.transform)

    _ = outputs.transform(dataframe)

    for output in outputs:
        output.transform.assert_called_once()
