"""Unit tests related to bench_mhc/utils/click/types.py."""

from pathlib import Path

import click
import pytest

from bench_mhc.utils.click.types import CommaSepOrFileToSet
from bench_mhc.utils.click.types import RelativePath
from bench_mhc.utils.io import save_txt


class TestRelativePath:
    """Test cases for the RelativePath class."""

    def test_relative_path(self, tmp_path: Path) -> None:
        """Ensure RelativePath properly converts a relative path to an absolute path."""
        relative_to_directory = tmp_path / "test_relative_path"
        relative_to_directory.mkdir()

        relative_path = relative_to_directory / "test_relative_path.txt"
        relative_path.touch()

        click_type = RelativePath(relative_to=relative_to_directory, strict=True)

        assert click_type.convert(relative_path.name, None, None) == str(relative_path)

    @pytest.mark.parametrize("strict", [True, False])
    @pytest.mark.parametrize("path_exists_in_working_dir", [True, False])
    def test_relative_path_non_existing_in_relative_to(
        self, tmp_path: Path, strict: bool, path_exists_in_working_dir: bool
    ) -> None:
        """Ensure RelativePath works when the path does not exist in the relative_to directory."""
        relative_to_directory = tmp_path / "test_relative_path"
        relative_to_directory.mkdir()

        working_directory = tmp_path / "working_directory"
        working_directory.mkdir()

        relative_path = working_directory / "test_relative_path.txt"
        if path_exists_in_working_dir:
            relative_path.touch()

        click_type = RelativePath(relative_to=relative_to_directory, strict=strict, exists=True)

        if strict:
            match_msg = (
                f"Issue with the provided path '{relative_path.name}' that should be relative to "
                f"'{relative_to_directory}': Path '{relative_to_directory / relative_path.name}' "
                "does not exist."
            )
            with pytest.raises(click.exceptions.BadParameter, match=match_msg):
                click_type.convert(relative_path.name, None, None)

        elif path_exists_in_working_dir:
            assert click_type.convert(relative_path, None, None) == str(relative_path)

        else:
            match_msg = (
                f"Issue with the provided path '{relative_path.name}': Path '{relative_path.name}' "
                f"does not exist. "
                f"Note that the path could not be used as relative to "
                f"'{relative_to_directory}': Path '{relative_to_directory / relative_path.name}' "
                "does not exist."
            )
            with pytest.raises(click.exceptions.BadParameter, match=match_msg):
                click_type.convert(relative_path.name, None, None)


class TestCommaSepOrFileToSet:
    """Test cases for the CommaSepOrFileToSet class."""

    @pytest.mark.parametrize(
        ("value", "item_click_type", "expected_values"),
        [
            ("11,14,16", None, {"11", "14", "16"}),
            ("11,14,16", click.INT, {11, 14, 16}),
            (
                "/path/to/one/file.csv,/path/to/another/file.csv",
                click.Path(exists=False),
                {"/path/to/one/file.csv", "/path/to/another/file.csv"},
            ),
        ],
    )
    def test_comma_sep_or_file_to_set(
        self, value: str, item_click_type: click.ParamType, expected_values: list
    ) -> None:
        """Ensure CommaSepOrFileToSet properly converts comma-separated set as input."""
        click_type = CommaSepOrFileToSet(item_click_type)

        assert click_type.convert(value, None, None) == expected_values

    def test_comma_sep_or_file_to_set_w_file(self, tmp_path: Path) -> None:
        """Ensure CommaSepOrFileToSet properly converts a file as input."""
        file_path = tmp_path / "test_comma_sep_or_file_to_set_w_file.txt"
        values = ["model1", "model2", "model3"]
        save_txt(values, file_path)

        click_type = CommaSepOrFileToSet()

        assert click_type.convert(str(file_path), None, None) == set(values)

    def test_comma_sep_or_file_to_set_non_existing_file(self) -> None:
        """Ensure a BadParameter is raised if one of the provided file paths does not exist."""
        click_type = CommaSepOrFileToSet(item_click_type=click.Path(exists=True))

        match_msg = (
            "Item '/path/to/one/file.csv' failed validation: "
            "Path '/path/to/one/file.csv' does not exist."
        )
        with pytest.raises(click.exceptions.BadParameter, match=match_msg):
            click_type.convert("/path/to/one/file.csv,/path/to/another/file.csv", None, None)
