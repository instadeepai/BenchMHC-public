"""Unit tests related to bench_mhc/utils/io.py."""

from pathlib import Path
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import patch

import polars as pl
import polars.testing as pl_testing
import pytest
import yaml

from bench_mhc.utils.io import JSONEncoderWPath
from bench_mhc.utils.io import load_fasta
from bench_mhc.utils.io import load_json
from bench_mhc.utils.io import load_txt
from bench_mhc.utils.io import load_yml
from bench_mhc.utils.io import save_json
from bench_mhc.utils.io import save_txt
from bench_mhc.utils.io import save_yml


@pytest.mark.parametrize("directory_does_not_exist", [False, True])
@pytest.mark.parametrize("strip_whitespaces", [False, True])
def test_save_load_txt(
    tmp_path: Path,
    strip_whitespaces: bool,
    directory_does_not_exist: bool,
) -> None:
    """Ensure we can save / load .txt files."""
    list_to_save = ["a ", " b", "c"]

    file_path = (
        tmp_path / "test_save_load_txt" / "test.txt"
        if directory_does_not_exist
        else tmp_path / "test.txt"
    )
    save_txt(list_to_save, file_path)

    expected_loaded_list = [line.strip() if strip_whitespaces else line for line in list_to_save]

    assert load_txt(file_path, strip_whitespaces=strip_whitespaces) == expected_loaded_list


@pytest.mark.parametrize("directory_does_not_exist", [False, True])
@patch("bench_mhc.utils.io.yaml.dump", wraps=yaml.dump)
def test_save_load_yml(
    yaml_dump_spy: MagicMock, directory_does_not_exist: bool, tmp_path: Path
) -> None:
    """Ensure we can save / load .yml / .yaml files."""
    mapping_to_save = {"a": 1, "b": 2, "c": 3}

    file_path = (
        tmp_path / "test_save_load_yml" / "test.yml"
        if directory_does_not_exist
        else tmp_path / "test.yml"
    )

    save_yml(mapping_to_save, file_path)

    yaml_dump_spy.assert_called_once_with(
        mapping_to_save, ANY, default_flow_style=False, sort_keys=False
    )

    assert load_yml(file_path) == mapping_to_save


@pytest.mark.parametrize("directory_does_not_exist", [False, True])
def test_save_load_json(directory_does_not_exist: bool, tmp_path: Path) -> None:
    """Ensure we can save / load .json files."""
    mapping: dict[str, int | Path | str] = {
        "a": 1,
        "b": 2,
        "c": 3,
        "path": Path("/path/test.csv"),
        "string_path": "/path/other/test.csv",
    }

    file_path = (
        tmp_path / "test_save_load_json" / "test.json"
        if directory_does_not_exist
        else tmp_path / "test.json"
    )
    save_json(mapping, file_path)

    expected_mapping = mapping.copy()
    expected_mapping["path"] = str(mapping["path"])

    assert load_json(file_path) == expected_mapping


def test_json_encoder_w_path() -> None:
    """Test the JSONEncoderWPath."""
    json_encoder_w_path = JSONEncoderWPath()

    mapping = {
        "path": Path("/path/test.csv"),
        "string_path": "/path/other/test.csv",
    }

    assert json_encoder_w_path.encode(mapping) == (
        '{"path": "/path/test.csv", "string_path": "/path/other/test.csv"}'
    )

    # Check we override only Path default as advised in JSONEncoder.default()
    match_msg = "Object of type str is not JSON serializable"
    with pytest.raises(TypeError, match=match_msg):
        json_encoder_w_path.default(mapping["string_path"])


def test_load_fasta(tmp_path: Path) -> None:
    """Test the load_fasta function."""
    data = {"id1": "SEQUENCE1", "id2": "SEQUENCE2", "id3": "SEQUENCE3"}
    fasta_file_path = tmp_path / "test.fasta"
    fasta_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(fasta_file_path, "w") as f:
        for seq_id, sequence in data.items():
            f.write(f">{seq_id}\n{sequence}\n")

    pl_testing.assert_frame_equal(
        load_fasta(fasta_file_path, columns=["id", "sequence"]),
        pl.from_records(list(data.items()), schema=["id", "sequence"]),
    )
