"""Module to implement the Dataset and DataModule classes."""

from collections.abc import Iterator
from pathlib import Path

import lightning
import numpy as np
import polars as pl
import torch
from polars import scan_csv
from torch.utils.data import DataLoader

from bench_mhc.dataset.cache import DatasetCache
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.io import load_json
from bench_mhc.utils.io import save_json
from bench_mhc.utils.logging import system
from bench_mhc.utils.mode import Mode
from bench_mhc.utils.stage import Stage
from bench_mhc.variables.variables import Outputs
from bench_mhc.variables.variables import Variables

log = system.get(__name__)


class Dataset(torch.utils.data.Dataset):
    """Generic dataset.

    Attributes:
        df: A Polars DataFrame containing the data.
        mode: The mode in which the dataset will be used (TRAIN, VAL, TEST, PREDICT).
        inputs: Variables corresponding to input features.
        outputs: Variables corresponding to output labels.
        inputs_data: Features' values for inputs.
        outputs_data: Features' values for outputs.
    """

    def __init__(self, df: pl.DataFrame, mode: Mode, inputs: Variables, outputs: Outputs) -> None:
        """Initialize the dataset.

        Args:
            df: A Polars DataFrame containing the data.
            mode: The mode in which the dataset will be used (TRAIN, VAL, TEST, PREDICT).
            inputs: Variables corresponding to input features.
            outputs: Variables corresponding to output labels.
        """
        super().__init__()

        self.df = df
        self.mode = mode

        self.inputs = inputs
        self.outputs = outputs

        self.inputs_data = {
            feature_name: np.array(feature_values)[:, np.newaxis]
            for feature_name, feature_values in df.select(self.inputs.features).to_dict().items()
        }

        self.outputs_data = {}
        if mode != Mode.PREDICT:
            self.outputs_data = {
                feature_name: np.array(feature_values)[:, np.newaxis]
                for feature_name, feature_values in df.select(self.outputs.features)
                .to_dict()
                .items()
            }

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | dict[str, np.ndarray]:
        """Return an item from the dataset.

        Args:
            index: The index of the item to return.

        Returns:
            If mode == 'PREDICT', a dictionary of the inputs.
            Else, a tuple containing the inputs' and outputs' dictionaries.
        """
        inputs = {
            feature_name: feature_values[index]
            for feature_name, feature_values in self.inputs_data.items()
        }

        if self.mode == Mode.PREDICT:
            return inputs
        else:
            outputs = {
                feature_name: feature_values[index]
                for feature_name, feature_values in self.outputs_data.items()
            }

            return inputs, outputs


class CustomDataLoader:
    """Custom DataLoader.

    This shuffles the data and generate batches of tensors when iterated upon.

    Using this custom DataLoader to feed data to the model on the GPU proved to be faster than
    using the native torch.utils.data.DataLoader with 8 workers, cf.
    https://github.com/instadeepai/BenchMHC/issues/61.

    Attributes:
        dataset: The Dataset object containing the data.
        inputs_data: Features' tensor values for inputs.
        outputs_data: Features' tensor values for outputs.
        batch_size: The batch size to use for each iteration.
        shuffle: Whether to shuffle the data.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool) -> None:
        """Initialize the custom data loader.

        Args:
            dataset: The Dataset object containing the data.
            batch_size: The batch size to use for each iteration.
            shuffle: Whether to shuffle the data.
        """
        self.dataset = dataset
        self.inputs_data = {
            input_name: torch.as_tensor(input_array)
            for input_name, input_array in dataset.inputs_data.items()
        }
        self.outputs_data = {
            output_name: torch.as_tensor(output_array)
            for output_name, output_array in dataset.outputs_data.items()
        }
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """Return the length of the data loader.

        Returns:
            The length of the data loader.
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(
        self,
    ) -> Iterator[
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | dict[str, torch.Tensor]
    ]:
        """Return an iterator for the data loader.

        Yields:
            If mode == 'PREDICT', a dictionary of the input tensors.
            Else, a tuple containing the input tensors' and output tensors' dictionaries.
        """
        if self.shuffle:
            indices = torch.randperm(len(self.dataset))
        else:
            indices = torch.arange(len(self.dataset))

        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            input_tensors = {
                feature_name: feature_values[batch_indices]
                for feature_name, feature_values in self.inputs_data.items()
            }

            if self.dataset.mode == Mode.PREDICT:
                yield input_tensors

            else:
                output_tensors = {
                    feature_name: feature_values[batch_indices]
                    for feature_name, feature_values in self.outputs_data.items()
                }

                yield input_tensors, output_tensors


class DataModule(lightning.LightningDataModule):
    """Generic data module.

    See https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule
    for the required methods.

    Attributes:
        mode2path: A mapping from mode to file path.
        mode2dataset: A mapping from mode to dataset.
        inputs: Variables corresponding to input features.
        outputs: Variables corresponding to output labels.
        batch_size: The batch size for data loaders.
        num_workers: The number of workers to use for data loading.
        prefetch_factor: The number of batches loaded in advance by each worker.
        _overridden_input_feature_name2value: Mapping from input feature name to value for inputs
            to be overridden.
        cache: An optional DatasetCache instance for caching transformed datasets if cache is used.
        mode2hash: A mapping from mode to hash of the transformed dataset.
        transformed_data_directory: The optional directory containing the transformed data, used
            if cache is not used.
    """

    def __init__(
        self,
        inputs: Variables,
        outputs: Outputs,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int | None,
        train_file_path: Path | None = None,
        val_file_path: Path | None = None,
        test_file_path: Path | None = None,
        predict_file_path: Path | None = None,
        transformed_data_directory: Path | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        """Initialize the data module.

        Args:
            inputs: Variables corresponding to input features.
            outputs: Variables corresponding to output labels.
            batch_size: The batch size for data loaders.
            num_workers: The number of workers to use for data loading.
            prefetch_factor: The number of batches loaded in advance by each worker.
            train_file_path: Optional path to the training data file.
            val_file_path: Optional path to the validation data file.
            test_file_path: Optional path to the test data file.
            predict_file_path: Optional path to the prediction data file.
            transformed_data_directory: Optional path to the directory where to save
                the transformed data.
            cache_directory: Optional path to the cache directory.$

        Raises:
            ValueError:
                - If 'transformed_data_directory' and 'cache_directory' are both provided.
                - If none of 'transformed_data_directory' and 'cache_directory' is provided.
        """
        super().__init__()

        if transformed_data_directory is None and cache_directory is None:
            raise ValueError(
                "Either 'transformed_data_directory' or 'cache_directory' must be provided."
            )
        if transformed_data_directory is not None and cache_directory is not None:
            raise ValueError(
                "Only one of 'transformed_data_directory' or 'cache_directory' must be provided."
            )

        self.transformed_data_directory = transformed_data_directory
        if self.transformed_data_directory is not None:
            self.transformed_data_directory.mkdir(parents=True, exist_ok=True)

        self.mode2path = {
            Mode.TRAIN: train_file_path,
            Mode.VAL: val_file_path,
            Mode.TEST: test_file_path,
            Mode.PREDICT: predict_file_path,
        }
        if all(path is None for path in self.mode2path.values()):
            log.warning(
                "No provided paths for DataModule initialization, transformed data will have "
                "to be saved manually for required modes with 'save_transformed_data'."
            )
        self.mode2dataset: dict[Mode, Dataset] = {}
        self.mode2hash: dict[Mode, str] = {}
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._overridden_input_feature_name2value: dict[str, np.ndarray] = {}

        self.cache = DatasetCache(cache_directory=cache_directory) if cache_directory else None

    def fit_variables(self) -> None:
        """Fit the variables on the training dataset.

        If a cache is available, the variables are loaded from the cache.
        Otherwise, the variables are fitted on the training dataset and saved to the cache.
        If the cache is not available, the variables are fitted on the training dataset
        and not saved anywhere.

        Raises:
            ValueError: If 'training_path' has not been provided to the DataModule.
        """
        training_path = self.mode2path[Mode.TRAIN]
        if training_path is None:
            raise ValueError(
                "Variables cannot be fit as 'training_path' is not provided to the DataModule."
            )

        lf = scan_csv(training_path)

        for variables_name in ["inputs", "outputs"]:
            variables = getattr(self, variables_name)
            if self.cache:
                hash_ = self.cache.compute_hash(lf, variables.to_dict())
                variables_cache_path = self.cache.cache_directory / f"{hash_}.json"

                if variables_cache_path.exists():
                    variables = (
                        Variables.from_dict(load_json(variables_cache_path))
                        if variables_name == "inputs"
                        else Outputs.from_dict(load_json(variables_cache_path))
                    )
                    log.info(
                        f"Loaded cached fitted {variables_name} from '{variables_cache_path}'."
                    )
                    setattr(self, variables_name, variables)

                else:
                    variables.fit(lf)
                    save_json(variables.to_dict(), variables_cache_path)
                    log.info(f"Saved fitted {variables_name} to '{variables_cache_path}'.")
            else:
                variables.fit(lf)

    def transformed_data_path(self, mode: Mode) -> Path:
        """Get the path to the transformed data.

        Args:
            mode: The mode to get the data from.

        Returns:
            The path to the transformed data.

        Raises:
            ValueError: If 'transformed_data_directory' is not provided to the DataModule.
        """
        if self.transformed_data_directory is None:
            raise ValueError("Transformed data directory is not provided to the DataModule.")

        return self.transformed_data_directory / f"{mode}.parquet"

    def save_transformed_data(self, df: pl.LazyFrame | pl.DataFrame, mode: Mode) -> None:
        """Save transformed data for specific mode.

        Args:
            df: LazyFrame containing the transformed data.
            mode: The mode to save the data to.
        """
        log.info(f"Saving transformed data for mode '{mode}'...")
        transformed_data_file_path = self.transformed_data_path(mode)

        if isinstance(df, pl.LazyFrame):
            df.sink_parquet(transformed_data_file_path)
        else:
            df.write_parquet(transformed_data_file_path)

        log.info(f"Transformed data for mode '{mode}' saved in '{transformed_data_file_path}'.")

    def prepare_data(self) -> None:
        """Transform and save datasets for each mode.

        Transforms the raw datasets into preprocessed datasets, fits variables if needed,
        and saves the transformed data to disk. If a cache is available, it will be used
        to avoid redundant transformations.

        Data is only transformed on provided file paths, cf. 'mode2path' attribute.

        Remarks:
            - With Lightning, 'prepare_data' is called from the main process. No state
            (e.g. self.x = y) is assigned here since it is called on a single process and this
            state won't be available for other processes.
        """
        log.info("Preparing datasets.")
        for mode, file_path in self.mode2path.items():
            if file_path is not None:
                if (
                    self.transformed_data_directory is not None
                    and self.transformed_data_path(mode).exists()
                ):
                    log.info(
                        f"Transformed dataset located in '{self.transformed_data_path(mode)}' "
                        f"for mode: '{mode}'."
                    )
                    continue

                log.info(f"Transforming dataset located in '{file_path}' for mode: '{mode}'.")
                variables = self.inputs if mode == Mode.PREDICT else self.inputs + self.outputs

                lf = pl.scan_csv(file_path)
                if mode == Mode.TRAIN and not variables.is_fitted:
                    log.info(f"Fitting variables on the dataset '{file_path}'.")
                    self.fit_variables()
                if self.cache:
                    metadata = {
                        "inputs": self.inputs.to_dict(),
                        "outputs": self.outputs.to_dict(),
                        "mode": mode.value,
                    }
                    hash_ = self.cache.compute_hash(lf, metadata)
                    self.mode2hash[mode] = hash_

                    if (self.cache.cache_directory / f"{hash_}.parquet").exists():
                        continue

                # If not in cache, transform and save
                transformed_lf = variables.transform(lf).select(variables.features)
                if self.cache:
                    self.cache.cache_from_hash(hash_, transformed_lf)
                else:
                    self.save_transformed_data(transformed_lf, mode)

    def _load_dataset(self, mode: Mode) -> Dataset:
        """Load a dataset for a specific mode.

        This method loads the transformed dataset either from the cache (if prepare_data has been
        called) or from the transformed directory (in case of iterative annotation process).
        If some input features to be overridden are attached to the DataModule,
        the dataset's input features will be overridden automatically.

        Args:
            mode: The mode of the dataset to load (TRAIN, VAL, TEST, PREDICT).

        Returns:
            A Dataset object for the specified mode.
        """
        if self.cache and self.mode2hash.get(mode) is not None:
            transformed_df = self.cache.load_from_hash(self.mode2hash[mode])
        else:
            file_path = self.transformed_data_path(mode)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Transformed data for mode '{mode}' not found in '{file_path}'."
                )
            transformed_df = pl.read_parquet(file_path)

        log.info(f"Loading transformed data for mode: '{mode}'.")
        dataset = Dataset(
            transformed_df,
            mode=mode,
            inputs=self.inputs,
            outputs=self.outputs,
        )

        self._override_input_features(dataset)

        return dataset

    def setup(self, stage: str) -> None:
        """Set up datasets for the given stage.

        Remarks:
            - With Lightning, 'setup' is called from every process across all the nodes.

        Args:
            stage: The stage for which to prepare datasets (FIT, TEST, PREDICT).

        Raises:
            ValueError: If an unknown stage is provided or required data is missing.
        """
        try:
            stage_enum = Stage(stage)
        except ValueError as error:
            raise ValueError(
                f"Unknown stage: {stage}. Possible values: {format_iterable(Stage.valid_values())}."
            ) from error

        if stage_enum == Stage.FIT:
            self.mode2dataset[Mode.TRAIN] = self._load_dataset(Mode.TRAIN)
            self.mode2dataset[Mode.VAL] = self._load_dataset(Mode.VAL)

        elif stage_enum == Stage.VALIDATE:
            self.mode2dataset[Mode.VAL] = self._load_dataset(Mode.VAL)

        elif stage_enum == Stage.TEST:
            self.mode2dataset[Mode.TEST] = self._load_dataset(Mode.TEST)

        elif stage_enum == Stage.PREDICT:
            self.mode2dataset[Mode.PREDICT] = self._load_dataset(Mode.PREDICT)

    def _get_dataloader(self, mode: Mode) -> DataLoader | CustomDataLoader:
        """Get the dataloader for the given mode.

        Args:
            mode: The mode of the dataset to load (TRAIN, VAL, TEST, PREDICT).

        Returns:
            Either a DataLoader or a CustomDataLoader.
        """
        if self.num_workers == 0:
            log.warning(
                f"'num_workers={self.num_workers}' provided. Data batches will be generated "
                "with CustomDataLoader instead of the native torch.utils.data.DataLoader to "
                "speed up the training."
            )
            return CustomDataLoader(
                dataset=self.mode2dataset[mode],
                batch_size=self.batch_size,
                shuffle=mode in {Mode.TRAIN, Mode.VAL},
            )

        log.info(
            f"'num_workers={self.num_workers}' provided. Data batches are generated with "
            "the native torch.utils.data.DataLoader."
        )
        return DataLoader(
            dataset=self.mode2dataset[mode],
            batch_size=self.batch_size,
            shuffle=mode in {Mode.TRAIN, Mode.VAL},
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader | CustomDataLoader:
        """Get the training DataLoader.

        Returns:
            A DataLoader object for training data.
        """
        return self._get_dataloader(mode=Mode.TRAIN)

    def val_dataloader(self) -> DataLoader | CustomDataLoader:
        """Get the validation DataLoader.

        Validation data is shuffled to avoid any inconsistencies between validation loss and
        metric, cf. #50.

        Returns:
            A DataLoader object for validation data.
        """
        return self._get_dataloader(mode=Mode.VAL)

    def test_dataloader(self) -> DataLoader | CustomDataLoader:
        """Get the test DataLoader.

        Returns:
            A DataLoader object for test data.
        """
        return self._get_dataloader(mode=Mode.TEST)

    def predict_dataloader(self) -> DataLoader | CustomDataLoader:
        """Get the prediction DataLoader.

        Returns:
            A DataLoader object for prediction data.
        """
        return self._get_dataloader(mode=Mode.PREDICT)

    @property
    def overridden_input_feature_name2value(self) -> dict[str, np.ndarray]:
        """Mapping from input feature name to value for inputs to be overridden.

        Returns:
            A mapping from input feature name to value for inputs to be overridden.
        """
        return self._overridden_input_feature_name2value

    @overridden_input_feature_name2value.setter
    def overridden_input_feature_name2value(
        self, input_feature_name2value: dict[str, np.ndarray]
    ) -> None:
        """Setter for the mapping from input feature name to value for inputs to be overridden.

        Args:
            input_feature_name2value: The mapping from input feature name to value.

        Raises:
            ValueError: If the feature to be overridden is not available in the dataset input
                features.
        """
        for feature_name in input_feature_name2value:
            if feature_name not in self.inputs.features:
                raise ValueError(
                    f"Feature '{feature_name}' to be overridden is not in the input "
                    f"features: {format_iterable(self.inputs.features)}."
                )

        self._overridden_input_feature_name2value = input_feature_name2value

    def _override_input_features(self, dataset: Dataset) -> None:
        """Override input features with one value.

        This is notably used to change allele values when using Dataset for the reference set
        in iterative annotation.

        Args:
            dataset: The dataset to override input features for.
        """
        for feature_name, feature_value in self.overridden_input_feature_name2value.items():
            feature_values = np.repeat(
                [feature_value], repeats=len(dataset.inputs_data[feature_name]), axis=0
            )
            dataset.inputs_data[feature_name] = feature_values
            log.info(f"Feature '{feature_name}' is overridden with value: {feature_value}.")
