"""Unit tests related to bench_mhc/model/mhc1_nn_align.py."""

from pathlib import Path
from typing import Any
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from bench_mhc.dataset.main import Dataset
from bench_mhc.model.mhc1_nn_align import MHC1NNAlignLightningModule
from bench_mhc.model.mhc1_nn_align import MHC1NNAlignModel
from bench_mhc.utils.format import format_iterable


@pytest.fixture
def batch(
    mhc1_nn_align_configuration: dict[str, dict[str, Any]], mhc1_nn_align_dataset: Dataset
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Batch of input and output tensors to test the model."""
    dataloader = DataLoader(
        mhc1_nn_align_dataset,
        batch_size=mhc1_nn_align_configuration["training"]["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    return next(iter(dataloader))


class TestMHC1NNAlignModel:
    """Test cases for the MHC1NNAlignModel class."""

    @pytest.mark.parametrize("hidden_dim_not_provided", [False, True])
    @pytest.mark.parametrize("wrong_input_names", [False, True])
    @pytest.mark.parametrize("inconsistent_vocabularies", [False, True])
    @patch("bench_mhc.model.mhc1_nn_align.random.choice", return_value=66)
    def test_init_mhc1_nn_align_model(
        self,
        random_choice_spy: MagicMock,
        mhc1_nn_align_configuration: dict[str, dict[str, Any]],
        mhc1_nn_align_dataset: Dataset,
        hidden_dim_not_provided: bool,
        wrong_input_names: bool,
        inconsistent_vocabularies: bool,
    ) -> None:
        """Test initializing the MHC1NNAlignModel class."""
        if hidden_dim_not_provided:
            mhc1_nn_align_configuration["model"].pop("hidden_dim")

        if wrong_input_names:
            mhc1_nn_align_dataset.inputs["peptide"].name = "wrong_peptide"

        if inconsistent_vocabularies:
            mhc1_nn_align_dataset.inputs["allele"]._aas = ["A", "C"]

        if hidden_dim_not_provided:
            match_msg = (
                "You must specify the dimension 'hidden_dim' of the hidden layer in the model "
                "configuration."
            )
            with pytest.raises(ValueError, match=match_msg):
                _ = MHC1NNAlignModel(
                    model_configuration=mhc1_nn_align_configuration["model"],
                    inputs=mhc1_nn_align_dataset.inputs,
                    outputs=mhc1_nn_align_dataset.outputs,
                )

        elif wrong_input_names:
            match_msg = (
                "'peptide' and 'allele' must be defined in inputs. Inputs provided: "
                f"{format_iterable(['wrong_peptide', 'allele'])}"
            )
            with pytest.raises(ValueError, match=match_msg):
                _ = MHC1NNAlignModel(
                    model_configuration=mhc1_nn_align_configuration["model"],
                    inputs=mhc1_nn_align_dataset.inputs,
                    outputs=mhc1_nn_align_dataset.outputs,
                )

        elif inconsistent_vocabularies:
            match_msg = "The vocabulary for the peptide and allele variables do not match."
            with pytest.raises(ValueError, match=match_msg):
                _ = MHC1NNAlignModel(
                    model_configuration=mhc1_nn_align_configuration["model"],
                    inputs=mhc1_nn_align_dataset.inputs,
                    outputs=mhc1_nn_align_dataset.outputs,
                )

        else:
            mhc1_nn_align_model = MHC1NNAlignModel(
                model_configuration=mhc1_nn_align_configuration["model"],
                inputs=mhc1_nn_align_dataset.inputs,
                outputs=mhc1_nn_align_dataset.outputs,
            )

            random_choice_spy.assert_called_once_with([66, 56])

            assert mhc1_nn_align_model.max_num_cores == 10

            assert isinstance(mhc1_nn_align_model.activation_layer, nn.Sigmoid)

            # Embedding layer
            expected_nb_non_trainable_parameters = 22 * 20

            # Hidden layer + 2 final layers
            input_size = 20 * (9 + 34) + 8
            expected_nb_trainable_parameters = 66 * (input_size + 1) + 2 * (1 * (66 + 1))

            nb_non_trainable_parameters = sum(
                p.numel() for p in mhc1_nn_align_model.parameters() if not p.requires_grad
            )
            nb_trainable_parameters = sum(
                p.numel() for p in mhc1_nn_align_model.parameters() if p.requires_grad
            )

            assert nb_non_trainable_parameters == expected_nb_non_trainable_parameters
            assert nb_trainable_parameters == expected_nb_trainable_parameters

    def test_forward_mhc1_nn_align_model(
        self,
        mhc1_nn_align_configuration: dict[str, dict[str, Any]],
        mhc1_nn_align_dataset: Dataset,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ) -> None:
        """Test the forward pass of the MHC1NNAlignModel class."""
        mhc1_nn_align_model = MHC1NNAlignModel(
            model_configuration=mhc1_nn_align_configuration["model"],
            inputs=mhc1_nn_align_dataset.inputs,
            outputs=mhc1_nn_align_dataset.outputs,
        )

        input_tensors, _ = batch
        output_tensors = mhc1_nn_align_model(input_tensors)

        for output in mhc1_nn_align_dataset.outputs:
            assert output.name in output_tensors.keys()

            output_tensor = output_tensors[output.name]
            assert output_tensor.shape == (mhc1_nn_align_configuration["training"]["batch_size"], 1)
            assert output_tensor.dtype == torch.float32


class TestMHC1NNAlignLightningModule:
    """Test cases for the MHC1NNAlignLightningModule class."""

    def test_mhc1_nn_align_lightning_module(
        self,
        model_directory: Path,
        mhc1_nn_align_configuration: dict[str, dict[str, Any]],
        mhc1_nn_align_dataset: Dataset,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ) -> None:
        """Test the MHC1NNAlignLightningModule class."""
        model_path = model_directory / "test_mhc1_nn_align_lightning_module"

        with patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory):
            module = MHC1NNAlignLightningModule(
                model_path=model_path,
                configuration=mhc1_nn_align_configuration,
                inputs=mhc1_nn_align_dataset.inputs,
                outputs=mhc1_nn_align_dataset.outputs,
            )

            assert module.model_path.exists()
            assert isinstance(module.model, MHC1NNAlignModel)

            assert module.hparams_path == model_path / "hparams.json"
            module.save_hparams()
            assert module.hparams_path.exists()

            input_tensors, label_tensors = batch
            output_tensors = module.forward(input_tensors)

            for output in mhc1_nn_align_dataset.outputs:
                assert output.name in output_tensors.keys()
                assert output_tensors[output.name].shape == label_tensors[output.name].shape

            with (
                patch.object(module, "log_dict") as log_dict_spy,
                patch.object(module, "log") as log_spy,
            ):
                loss = module.training_step(batch)
                assert not loss.isnan()
                log_spy.assert_called_once_with(
                    "train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True
                )
                module.on_train_epoch_end()

                module.validation_step(batch)
                module.on_validation_epoch_end()

                assert log_dict_spy.call_count == 6
                log_dict_spy.assert_has_calls(
                    [
                        call(
                            {
                                "step": 0,
                                "train/hit/loss": ANY,
                                "train/binding_affinity/loss": ANY,
                                "train/loss": loss,
                            },
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                        ),
                        call(
                            {
                                "train/hit/average_precision": ANY,
                                "train/hit/binary_accuracy": ANY,
                                "step": 0,
                            },
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                        ),
                        call(
                            {"train/binding_affinity/mean_squared_error": ANY, "step": 0},
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                        ),
                        call(
                            {
                                "step": 0,
                                "val/hit/loss": ANY,
                                "val/binding_affinity/loss": ANY,
                                "val/loss": loss,
                            },
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                        ),
                        call(
                            {
                                "val/hit/average_precision": ANY,
                                "val/hit/binary_accuracy": ANY,
                                "step": 0,
                            },
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                        ),
                        call(
                            {
                                "step": 0,
                                "val/binding_affinity/mean_squared_error": ANY,
                            },
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                        ),
                    ]
                )

    @pytest.mark.parametrize("model_path", ["test_model", "sub_model_dir/test_model"])
    def test_mhc1_nn_align_lightning_module_relative_model_path(
        self,
        model_directory: Path,
        mhc1_nn_align_configuration: dict[str, dict[str, Any]],
        mhc1_nn_align_dataset: Dataset,
        model_path: str,
    ) -> None:
        """Test initializing MHC1NNAlignLightningModule with paths relative to MODEL_DIRECTORY."""
        with patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory):
            module = MHC1NNAlignLightningModule(
                model_path=Path(model_path),
                configuration=mhc1_nn_align_configuration,
                inputs=mhc1_nn_align_dataset.inputs,
                outputs=mhc1_nn_align_dataset.outputs,
            )

            assert module.model_path == model_directory / model_path
            assert module.model_path.exists()
            assert str(module.hparams["model_path"]) == model_path

    def test_mhc1_nn_align_lightning_module_model_path_not_in_models_dir(
        self,
        model_directory: Path,
        mhc1_nn_align_configuration: dict[str, dict[str, Any]],
        mhc1_nn_align_dataset: Dataset,
    ) -> None:
        """Test MHC1NNAlignLightningModule cannot be initialized if path not in MODEL_DIRECTORY."""
        model_path = Path("/path/not/in/models/dir/test_model")
        with patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory):
            match_msg = (
                f"The provided path {model_path} is not relative to "
                f"MODEL_DIRECTORY={model_directory}."
            )
            with pytest.raises(ValueError, match=match_msg):
                _ = MHC1NNAlignLightningModule(
                    model_path=Path(model_path),
                    configuration=mhc1_nn_align_configuration,
                    inputs=mhc1_nn_align_dataset.inputs,
                    outputs=mhc1_nn_align_dataset.outputs,
                )

    @pytest.mark.parametrize(
        ("optimizer_config", "expected_type", "expected_params"),
        [
            (
                {
                    "class_name": "SGD",
                    "lr": 0.05,
                },
                torch.optim.SGD,
                {"lr": 0.05},
            ),
            (
                {
                    "class_name": "Adam",
                    "lr": 0.001,
                    "weight_decay": 0.01,
                },
                torch.optim.Adam,
                {"lr": 0.001, "weight_decay": 0.01},
            ),
            (
                {
                    "class_name": "RMSprop",
                    "lr": 0.01,
                    "alpha": 0.99,
                    "eps": 1e-8,
                },
                torch.optim.RMSprop,
                {"lr": 0.01, "alpha": 0.99, "eps": 1e-8},
            ),
            (
                {
                    "class_name": "Adam",
                },
                torch.optim.Adam,
                {"lr": 0.001},  # default Adam lr
            ),
            (
                {
                    "class_name": "SGD",
                    "lr": 0.1,
                    "momentum": 0.9,
                },
                torch.optim.SGD,
                {"lr": 0.1, "momentum": 0.9},
            ),
        ],
    )
    def test_configure_optimizers(
        self,
        model_directory: Path,
        mhc1_nn_align_configuration: dict[str, dict[str, Any]],
        mhc1_nn_align_dataset: Dataset,
        optimizer_config: dict[str, Any],
        expected_type: type,
        expected_params: dict[str, Any],
    ) -> None:
        """Test the configure_optimizers function of the MHC1NNAlignLightningModule class."""
        model_path = model_directory / "test_configure_optimizers"
        mhc1_nn_align_configuration["training"]["optimizer"] = optimizer_config

        with patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory):
            module = MHC1NNAlignLightningModule(
                model_path=model_path,
                configuration=mhc1_nn_align_configuration,
                inputs=mhc1_nn_align_dataset.inputs,
                outputs=mhc1_nn_align_dataset.outputs,
            )

            optimizer = module.configure_optimizers()
            assert isinstance(optimizer, expected_type)

            for param_name, expected_value in expected_params.items():
                assert optimizer.param_groups[0][param_name] == expected_value

    @pytest.mark.parametrize(
        ("optimizer_config", "expected_error", "error_match"),
        [
            (
                {
                    "class_name": "NonExistentOptimizer",
                },
                AttributeError,
                None,
            ),
            (
                {
                    "class_name": "Adam",
                    "lr": "invalid_lr",
                },
                Exception,
                "Error instantiating optimizer",
            ),
        ],
    )
    def test_configure_optimizers_error_cases(
        self,
        model_directory: Path,
        mhc1_nn_align_configuration: dict[str, dict[str, Any]],
        mhc1_nn_align_dataset: Dataset,
        optimizer_config: dict[str, Any] | str,
        expected_error: type[Exception],
        error_match: str | None,
    ) -> None:
        """Test error cases for the configure_optimizers function."""
        model_path = model_directory / "test_configure_optimizers"
        mhc1_nn_align_configuration["training"]["optimizer"] = optimizer_config

        with patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory):
            module = MHC1NNAlignLightningModule(
                model_path=model_path,
                configuration=mhc1_nn_align_configuration,
                inputs=mhc1_nn_align_dataset.inputs,
                outputs=mhc1_nn_align_dataset.outputs,
            )

            if error_match:
                with pytest.raises(expected_error, match=error_match):
                    module.configure_optimizers()
            else:
                with pytest.raises(expected_error):
                    module.configure_optimizers()
