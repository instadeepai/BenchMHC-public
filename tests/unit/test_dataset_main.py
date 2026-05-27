"""Unit tests related to bench_mhc/dataset/main.py."""

from pathlib import Path
from typing import Any
from unittest.mock import call
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import torch
from polars.datatypes.classes import NumericType
from torch.utils.data import DataLoader

from bench_mhc.dataset.cache import DatasetCache
from bench_mhc.dataset.main import CustomDataLoader
from bench_mhc.dataset.main import DataModule
from bench_mhc.dataset.main import Dataset
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.io import load_json
from bench_mhc.utils.mode import Mode
from bench_mhc.utils.stage import Stage
from bench_mhc.variables.variables import Outputs
from bench_mhc.variables.variables import Variables

PL_TYPE2NP_TYPE = {
    pl.UInt8: np.uint8,
    pl.Int32: np.int32,
    pl.Float32: np.float32,
}

PL_TYPE2TORCH_TYPE = {
    pl.UInt8: torch.uint8,
    pl.Int32: torch.int32,
    pl.Float32: torch.float32,
}


@pytest.fixture
def inputs(variables_configuration: dict[str, dict[str, Any]]) -> Variables:
    """Input variables for the tests."""
    return Variables.from_dict(variables_configuration["inputs"])


@pytest.fixture
def overridden_input_feature_name2value(inputs: Variables) -> dict[str, np.ndarray]:
    """Mapping from variable name to value for the inputs to be overridden for the tests."""
    return {"numeric": np.array([42.0], dtype=PL_TYPE2NP_TYPE[inputs["numeric"].polars_type])}


@pytest.fixture
def outputs(variables_configuration: dict[str, dict[str, Any]]) -> Outputs:
    """Output variables for the tests."""
    return Outputs.from_dict(variables_configuration["outputs"])


def check_inputs_and_outputs(
    dataset_or_dataloader: Dataset | DataLoader | CustomDataLoader,
    mode: Mode,
    batch_size: int,
    inputs: Variables,
    outputs: Outputs,
    pl_type_mapping: dict[type[NumericType], Any],
) -> None:
    """Check inputs and outputs when iterating over dataloader or dataset."""
    if mode != Mode.PREDICT:
        item_: tuple[dict[str, torch.Tensor | np.ndarray], dict[str, torch.Tensor | np.ndarray]] = (
            next(iter(dataset_or_dataloader))  # type: ignore
        )
        model_inputs, model_outputs = item_[0], item_[1]

        assert outputs.features == set(model_outputs.keys())

        for output_variable in outputs.variables:
            for feature in output_variable.features:
                assert model_outputs[feature].dtype == pl_type_mapping[output_variable.polars_type]
                assert len(model_outputs[feature]) == batch_size

    else:
        model_inputs: dict[str, torch.Tensor | np.ndarray] = next(iter(dataset_or_dataloader))  # type: ignore

    assert inputs.features == set(model_inputs.keys())

    for input_variable in inputs.variables:
        for feature in input_variable.features:
            assert model_inputs[feature].dtype == pl_type_mapping[input_variable.polars_type]
            assert len(model_inputs[feature]) == batch_size


@pytest.mark.parametrize("mode", [Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT])
def test_dataset(dataframe: pl.DataFrame, inputs: Variables, outputs: Outputs, mode: Mode) -> None:
    """Test the Dataset class."""
    variables = inputs + outputs

    variables.fit(dataframe)

    transformed_dataframe = variables.transform(dataframe)

    dataset = Dataset(df=transformed_dataframe, mode=mode, inputs=inputs, outputs=outputs)

    assert len(dataset) == len(dataframe)

    check_inputs_and_outputs(
        dataset_or_dataloader=dataset,
        mode=mode,
        inputs=inputs,
        outputs=outputs,
        batch_size=1,
        pl_type_mapping=PL_TYPE2NP_TYPE,
    )


@pytest.mark.parametrize("mode", [Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT])
@pytest.mark.parametrize("shuffle", [False, True])
def test_custom_data_loader(
    dataframe: pl.DataFrame, inputs: Variables, outputs: Outputs, mode: Mode, shuffle: bool
) -> None:
    """Test the CustomDataLoader class.

    We check the data is shuffled when iterating over dataloader and shuffle=True.
    """
    variables = inputs + outputs
    variables.fit(dataframe)
    transformed_dataframe = variables.transform(dataframe)

    dataset = Dataset(df=transformed_dataframe, mode=mode, inputs=inputs, outputs=outputs)

    batch_size = 2
    custom_data_loader = CustomDataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    assert len(custom_data_loader) == ((len(dataframe) + batch_size - 1) // batch_size)

    for input_tensor in custom_data_loader.inputs_data.values():
        assert isinstance(input_tensor, torch.Tensor)
        assert len(input_tensor) == dataframe.select(pl.len()).item()

    for output_tensor in custom_data_loader.outputs_data.values():
        assert isinstance(output_tensor, torch.Tensor)
        assert len(output_tensor) == dataframe.select(pl.len()).item()

    with (
        patch("bench_mhc.dataset.main.torch.randperm", wraps=torch.randperm) as randperm_spy,
    ):
        check_inputs_and_outputs(
            dataset_or_dataloader=custom_data_loader,
            mode=mode,
            inputs=inputs,
            outputs=outputs,
            batch_size=batch_size,
            pl_type_mapping=PL_TYPE2TORCH_TYPE,
        )

    if shuffle:
        randperm_spy.assert_called_once_with(len(dataframe))
    else:
        randperm_spy.assert_not_called()


@pytest.fixture
def mode2path(
    variables_configuration: dict[str, dict[str, Any]], dataframe: pl.DataFrame, tmp_path: Path
) -> dict[str, Path]:
    """File paths for train / val / test / predict modes."""
    mode2file_path = {}
    for mode in Mode.valid_values():
        columns = {
            input_configuration.get("column") or input_name
            for input_name, input_configuration in variables_configuration["inputs"].items()
        }
        if mode in [Mode.TRAIN, Mode.VAL, Mode.TEST]:
            columns = columns.union(
                {
                    output_configuration.get("column") or output_name
                    for output_name, output_configuration in variables_configuration[
                        "outputs"
                    ].items()
                }
            )

        file_path = tmp_path / f"{mode}.csv"

        dataframe.select(columns).write_csv(file_path)

        mode2file_path[mode] = file_path

    return mode2file_path


@pytest.fixture
def transformed_data_directory(tmp_path: Path) -> Path:
    """Transformed data directory for tests."""
    transformed_data_directory = tmp_path / "transformed_data_directory"

    return transformed_data_directory


class TestDataModule:
    """Test cases for the DataModule class."""

    @pytest.mark.parametrize("num_workers", [0, 2])
    @pytest.mark.parametrize("stage", [Stage.FIT, Stage.VALIDATE, Stage.PREDICT, Stage.TEST])
    @pytest.mark.parametrize(
        ("initialize_with_file_paths", "transformed_dataframe_type"),
        [(False, "lazy"), (False, "eager"), (True, "lazy")],
    )
    def test_data_module(
        self,
        inputs: Variables,
        outputs: Outputs,
        transformed_data_directory: Path,
        mode2path: dict[str, Path],
        num_workers: int,
        stage: Stage,
        initialize_with_file_paths: bool,
        transformed_dataframe_type: str,
        overridden_input_feature_name2value: dict[str, np.ndarray],
    ) -> None:
        """Test the DataModule class.

        We check that:
        - the class can be instantiated,
        - the data module cannot be setup if the data is not prepared,
        - the transformed data can be prepared:
          - either from datasets (with `prepare_data`, if the DataModule is initialized with file
            paths),
          - or manually by using `save_transformed_data`,
        - the data module can be setup once the data has been prepared,
        - the dataloaders are instantiated with CustomDataLoader when num_workers=0,
        - the dataloaders are properly configured,
        - the values of some inputs can be overridden,
        - the dataloader can be iterated with proper inputs and outputs.
        """
        batch_size = 2
        data_module = DataModule(
            transformed_data_directory=transformed_data_directory,
            inputs=inputs,
            outputs=outputs,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers != 0 else None,
            train_file_path=mode2path["train"] if initialize_with_file_paths else None,
            val_file_path=mode2path["val"] if initialize_with_file_paths else None,
            test_file_path=mode2path["test"] if initialize_with_file_paths else None,
            predict_file_path=mode2path["predict"] if initialize_with_file_paths else None,
        )
        data_module.overridden_input_feature_name2value = overridden_input_feature_name2value

        if initialize_with_file_paths:
            for mode, path in data_module.mode2path.items():
                assert path == mode2path[mode.value]

        stage2mode = {
            Stage.FIT: Mode.TRAIN,
            Stage.VALIDATE: Mode.VAL,
            Stage.PREDICT: Mode.PREDICT,
            Stage.TEST: Mode.TEST,
        }
        match_msg = f"mode '{stage2mode[stage]}' not found"
        with pytest.raises(FileNotFoundError, match=match_msg):
            data_module.setup(stage=stage)

        assert not inputs.is_fitted
        assert not outputs.is_fitted

        if initialize_with_file_paths:
            data_module.prepare_data()

        else:
            for mode_value, file_path in mode2path.items():
                mode = Mode(mode_value)
                variables = inputs if mode == Mode.PREDICT else inputs + outputs

                lf = pl.scan_csv(file_path)
                if mode == Mode.TRAIN and not variables.is_fitted:
                    variables.fit(lf)

                transformed_lf = variables.transform(lf).select(variables.features)
                data_module.save_transformed_data(
                    transformed_lf
                    if transformed_dataframe_type == "lazy"
                    else transformed_lf.collect(),
                    mode,
                )

        for mode in [Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT]:
            assert (
                data_module.transformed_data_directory / f"{mode}.parquet"  # type: ignore
                in data_module.transformed_data_directory.iterdir()  # type: ignore
            )

        assert inputs.is_fitted
        assert outputs.is_fitted  # type: ignore

        data_module.setup(stage=stage)

        with (
            patch("bench_mhc.dataset.main.DataLoader", wraps=DataLoader) as data_loader_spy,
            patch(
                "bench_mhc.dataset.main.CustomDataLoader", wraps=DataLoader
            ) as custom_data_loader_spy,
        ):
            if stage == Stage.FIT:
                assert Mode.TRAIN in data_module.mode2dataset
                assert Mode.VAL in data_module.mode2dataset

                train_dataloader = data_module.train_dataloader()
                val_dataloader = data_module.val_dataloader()

                if num_workers == 0:
                    assert custom_data_loader_spy.call_count == 2
                    custom_data_loader_spy.assert_has_calls(
                        [
                            call(
                                dataset=data_module.mode2dataset[Mode.TRAIN],
                                batch_size=batch_size,
                                shuffle=True,
                            ),
                            call(
                                dataset=data_module.mode2dataset[Mode.VAL],
                                batch_size=batch_size,
                                shuffle=True,
                            ),
                        ]
                    )

                else:
                    assert data_loader_spy.call_count == 2
                    data_loader_spy.assert_has_calls(
                        [
                            call(
                                dataset=data_module.mode2dataset[Mode.TRAIN],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2,
                                prefetch_factor=2,
                                persistent_workers=True,
                                pin_memory=True,
                            ),
                            call(
                                dataset=data_module.mode2dataset[Mode.VAL],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2,
                                prefetch_factor=2,
                                persistent_workers=True,
                                pin_memory=True,
                            ),
                        ]
                    )

                check_inputs_and_outputs(
                    dataset_or_dataloader=train_dataloader,
                    mode=stage2mode[stage],
                    inputs=inputs,
                    outputs=outputs,
                    batch_size=batch_size,
                    pl_type_mapping=PL_TYPE2TORCH_TYPE,
                )
                check_inputs_and_outputs(
                    dataset_or_dataloader=val_dataloader,
                    mode=stage2mode[stage],
                    inputs=inputs,
                    outputs=outputs,
                    batch_size=batch_size,
                    pl_type_mapping=PL_TYPE2TORCH_TYPE,
                )

                # Assert 'numeric' is properly overridden
                assert torch.all(next(iter(train_dataloader))[0]["numeric"] == 42.0)
                assert torch.all(next(iter(val_dataloader))[0]["numeric"] == 42.0)

            elif stage == Stage.VALIDATE:
                assert Mode.VAL in data_module.mode2dataset

                val_dataloader = data_module.val_dataloader()

                if num_workers == 0:
                    custom_data_loader_spy.assert_called_once_with(
                        dataset=data_module.mode2dataset[Mode.VAL],
                        batch_size=batch_size,
                        shuffle=True,
                    )
                else:
                    data_loader_spy.assert_called_once_with(
                        dataset=data_module.mode2dataset[Mode.VAL],
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=2,
                        prefetch_factor=2,
                        persistent_workers=True,
                        pin_memory=True,
                    )

                check_inputs_and_outputs(
                    dataset_or_dataloader=val_dataloader,
                    mode=stage2mode[stage],
                    inputs=inputs,
                    outputs=outputs,
                    batch_size=batch_size,
                    pl_type_mapping=PL_TYPE2TORCH_TYPE,
                )

                # Assert 'numeric' is properly overridden
                assert torch.all(next(iter(val_dataloader))[0]["numeric"] == 42.0)

            elif stage == Stage.PREDICT:
                assert Mode.PREDICT in data_module.mode2dataset

                predict_dataloader = data_module.predict_dataloader()

                if num_workers == 0:
                    custom_data_loader_spy.assert_called_once_with(
                        dataset=data_module.mode2dataset[Mode.PREDICT],
                        batch_size=batch_size,
                        shuffle=False,
                    )
                else:
                    data_loader_spy.assert_called_once_with(
                        dataset=data_module.mode2dataset[Mode.PREDICT],
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2,
                        prefetch_factor=2,
                        persistent_workers=True,
                        pin_memory=True,
                    )

                check_inputs_and_outputs(
                    dataset_or_dataloader=predict_dataloader,
                    mode=stage2mode[stage],
                    inputs=inputs,
                    outputs=outputs,
                    batch_size=batch_size,
                    pl_type_mapping=PL_TYPE2TORCH_TYPE,
                )

                # Assert 'numeric' is properly overridden
                assert torch.all(next(iter(predict_dataloader))["numeric"] == 42.0)

            else:
                assert Mode.TEST in data_module.mode2dataset

                test_dataloader = data_module.test_dataloader()

                if num_workers == 0:
                    custom_data_loader_spy.assert_called_once_with(
                        dataset=data_module.mode2dataset[Mode.TEST],
                        batch_size=batch_size,
                        shuffle=False,
                    )
                else:
                    data_loader_spy.assert_called_once_with(
                        dataset=data_module.mode2dataset[Mode.TEST],
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2,
                        prefetch_factor=2,
                        persistent_workers=True,
                        pin_memory=True,
                    )

                check_inputs_and_outputs(
                    dataset_or_dataloader=test_dataloader,
                    mode=stage2mode[stage],
                    inputs=inputs,
                    outputs=outputs,
                    batch_size=batch_size,
                    pl_type_mapping=PL_TYPE2TORCH_TYPE,
                )

                # Assert 'numeric' is properly overridden
                assert torch.all(next(iter(test_dataloader))[0]["numeric"] == 42.0)

    @pytest.mark.parametrize("cache_directory_provided", [False, True])
    @pytest.mark.parametrize("train_file_path_provided", [False, True])
    def test_data_module_fit_variables(
        self,
        inputs: Variables,
        outputs: Outputs,
        transformed_data_directory: Path,
        mode2path: dict[str, Path],
        tmp_path: Path,
        train_file_path_provided: bool,
        cache_directory_provided: bool,
    ) -> None:
        """Test DataModule.fit_variables."""
        data_module = DataModule(
            transformed_data_directory=transformed_data_directory
            if not cache_directory_provided
            else None,
            inputs=inputs,
            outputs=outputs,
            batch_size=2,
            num_workers=2,
            prefetch_factor=2,
            train_file_path=None if not train_file_path_provided else mode2path["train"],
            val_file_path=mode2path["val"],
            test_file_path=mode2path["test"],
            predict_file_path=mode2path["predict"],
            cache_directory=tmp_path if cache_directory_provided else None,
        )

        assert not inputs.is_fitted
        assert not outputs.is_fitted

        if train_file_path_provided:
            with patch(
                "bench_mhc.dataset.main.DatasetCache.compute_hash",
                side_effect=["inputs_hash", "outputs_hash"],
            ) as compute_hash_spy:
                data_module.fit_variables()

            assert inputs.is_fitted
            assert outputs.is_fitted  # type: ignore

            if cache_directory_provided:
                assert compute_hash_spy.call_count == 2
                assert load_json(tmp_path / "inputs_hash.json") == inputs.to_dict()
                assert load_json(tmp_path / "outputs_hash.json") == outputs.to_dict()

                with (
                    patch(
                        "bench_mhc.dataset.main.load_json", side_effect=load_json
                    ) as load_json_spy,
                    patch(
                        "bench_mhc.dataset.main.DatasetCache.compute_hash",
                        side_effect=["inputs_hash", "outputs_hash"],
                    ) as compute_hash_spy,
                ):
                    data_module.fit_variables()
                    load_json_spy.assert_has_calls(
                        [call(tmp_path / "inputs_hash.json"), call(tmp_path / "outputs_hash.json")]
                    )

        else:
            match_msg = (
                "Variables cannot be fit as 'training_path' is not provided to the DataModule."
            )
            with pytest.raises(ValueError, match=match_msg):
                data_module.fit_variables()

    def test_data_module_unknown_stage(
        self,
        inputs: Variables,
        outputs: Outputs,
        transformed_data_directory: Path,
        mode2path: dict[str, Path],
    ) -> None:
        """Test DataModule.setup raises a ValueError with unknown stage."""
        data_module = DataModule(
            transformed_data_directory=transformed_data_directory,
            inputs=inputs,
            outputs=outputs,
            batch_size=2,
            num_workers=2,
            prefetch_factor=2,
            train_file_path=mode2path["train"],
            val_file_path=mode2path["val"],
            test_file_path=mode2path["test"],
            predict_file_path=mode2path["predict"],
        )

        match_msg = (
            "Unknown stage: unknown_stage. Possible values: "
            f"{format_iterable(Stage.valid_values())}."
        )
        with pytest.raises(ValueError, match=match_msg):
            data_module.setup(stage="unknown_stage")

    def test_data_module_unknown_feature_to_override(
        self,
        inputs: Variables,
        outputs: Outputs,
        transformed_data_directory: Path,
        mode2path: dict[str, Path],
        overridden_input_feature_name2value: dict[str, np.ndarray],
    ) -> None:
        """Test DataModule.overridden_input_feature_name2value breaks if feature not present."""
        inputs_dict = inputs.to_dict()
        inputs_dict.pop("numeric")
        inputs_wo_numeric = Variables.from_dict(inputs_dict)

        data_module = DataModule(
            transformed_data_directory=transformed_data_directory,
            inputs=inputs_wo_numeric,
            outputs=outputs,
            batch_size=2,
            num_workers=2,
            prefetch_factor=2,
            train_file_path=mode2path["train"],
            val_file_path=mode2path["val"],
            test_file_path=mode2path["test"],
            predict_file_path=mode2path["predict"],
        )

        match_msg = (
            f"Feature 'numeric' to be overridden is not in the input "
            f"features: {format_iterable(inputs_wo_numeric.features)}."
        )
        with pytest.raises(ValueError, match=match_msg):
            data_module.overridden_input_feature_name2value = overridden_input_feature_name2value

    def test_data_module_cache_functionality(
        self,
        inputs: Variables,
        outputs: Outputs,
        mode2path: dict[str, Path],
        tmp_path: Path,
        cache_directory: Path,
    ) -> None:
        """Test that the cache is properly used.

        We check that:
        - cache files are created during prepare_data,
        - cache is used for loading data,
        - an error is raised when cache is provided if the cache file is not found
            at the setup stage.
        """
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        data_module = DataModule(
            inputs=inputs,
            outputs=outputs,
            batch_size=2,
            num_workers=2,
            prefetch_factor=2,
            train_file_path=mode2path["train"],
            val_file_path=mode2path["val"],
            test_file_path=mode2path["test"],
            predict_file_path=mode2path["predict"],
            cache_directory=cache_directory,
        )

        # Check cache initialization
        assert isinstance(data_module.cache, DatasetCache)
        assert data_module.cache.cache_directory == cache_directory
        assert data_module.transformed_data_directory is None

        # Prepare data and check cache files are created
        modes = [Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT]
        hash_values = [
            "train_inputs_hash",
            "train_outputs_hash",
            "train_hash",
            "val_hash",
            "test_hash",
            "predict_hash",
        ]
        with patch(
            "bench_mhc.dataset.main.DatasetCache.compute_hash",
            side_effect=hash_values,
        ) as compute_hash_spy:
            data_module.prepare_data()

        # Called one time for each dataset mode
        # Called one time for train inputs and outputs
        assert compute_hash_spy.call_count == 6

        for mode in modes:
            assert (cache_directory / f"{mode.value}_hash.parquet").exists()
            assert mode in data_module.mode2hash

        assert (cache_directory / "train_inputs_hash.json").exists()
        assert (cache_directory / "train_outputs_hash.json").exists()

        with patch("bench_mhc.dataset.main.DatasetCache.load_from_hash") as load_from_hash_spy:
            data_module.setup(stage=Stage.PREDICT)
            load_from_hash_spy.assert_has_calls([call("predict_hash")])

        Path(cache_directory / "predict_hash.parquet").unlink()
        with pytest.raises(
            FileNotFoundError, match="Dataset with hash predict_hash not found in cache."
        ):
            data_module.setup(stage=Stage.PREDICT)

    def test_data_module_cache_and_transformed_data_directory(
        self,
        inputs: Variables,
        outputs: Outputs,
    ) -> None:
        """Test DataModule argument validation for data directories.

        Tests that DataModule properly validates the cache_directory and
        transformed_data_directory arguments:
        - Raises ValueError if both directories are None
        - Raises ValueError if both directories are provided (must specify only one)
        - Raises ValueError if transformed directory is not provided
            and transformed_data_path is called.
        """
        with pytest.raises(
            ValueError,
            match="Either 'transformed_data_directory' or 'cache_directory' must be provided",
        ):
            DataModule(
                inputs=inputs,
                outputs=outputs,
                batch_size=2,
                num_workers=2,
                prefetch_factor=2,
                cache_directory=None,
                transformed_data_directory=None,
            )

        with pytest.raises(
            ValueError,
            match="Only one of 'transformed_data_directory' or 'cache_directory' must be provided",
        ):
            DataModule(
                inputs=inputs,
                outputs=outputs,
                batch_size=2,
                num_workers=2,
                prefetch_factor=2,
                cache_directory=Path("dummy_cache_directory"),
                transformed_data_directory=Path("dummy_transformed_data_directory"),
            )

        with pytest.raises(ValueError, match="Transformed data directory is not provided"):
            DataModule(
                inputs=inputs,
                outputs=outputs,
                batch_size=2,
                num_workers=2,
                prefetch_factor=2,
                cache_directory=Path("dummy_cache_directory"),
            ).transformed_data_path(Mode.TRAIN)
