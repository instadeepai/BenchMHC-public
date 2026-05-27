"""Module to implement the MHC1 NNAlign model."""

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from bench_mhc.constants import MODEL_DIRECTORY
from bench_mhc.custom_objects.layers import Blosum62Embedding
from bench_mhc.model.base import BaseLightningModule
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.io import save_json
from bench_mhc.utils.logging import system
from bench_mhc.variables.variables import Outputs
from bench_mhc.variables.variables import Variables

log = system.get(__name__)


class MHC1NNAlignModel(nn.Module):
    """MHC1NNAlignModel PyTorch model (base architecture for NetMHCpan-4.1).

    Whenever per-epoch metric / loss is logged, we use `self.log_dict` instead of `self.log`
    to be able to pass the epoch as the step and allow correct x-axis for the logger, cf.
    https://github.com/Lightning-AI/pytorch-lightning/issues/3228#issuecomment-1011130151.

    Attributes:
        hidden_dim: Hidden dimension of the model.
        max_num_cores: Maximum number of 9mer cores to use in the architecture.
        embedding_layer: Embedding layer to use for AA sequences.
        flatten_layer: The flatten layer.
        core_hidden_layer: The Linear layer applied to a given core and its features.
        outputs: Outputs.
        output2final_layer: A mapping from output name to final layer on top of hidden vector.
        activation_layer: Activation layer.
    """

    def __init__(
        self,
        model_configuration: dict[str, Any],
        inputs: Variables,
        outputs: Outputs,
    ) -> None:
        """Initialize the NNAlign PyTorch model.

        Args:
            model_configuration: Model configuration.
            inputs: Input variables.
            outputs: Output variables.

        Raises:
            ValueError: If hidden_dim is not set in model configuration.
            ValueError: If inputs are not 'peptide' and 'allele.
            ValueError: If the vocabularies for 'peptide' and 'allele' do not match.
        """
        super().__init__()

        try:
            self.hidden_dim = model_configuration["hidden_dim"]
            if self.hidden_dim is None:
                self.hidden_dim = random.choice([66, 56])
                model_configuration["hidden_dim"] = self.hidden_dim
                log.info(f"'hidden_dim' is set to {self.hidden_dim}.")
        except KeyError as error:
            raise ValueError(
                "You must specify the dimension 'hidden_dim' of the hidden layer in the model "
                "configuration."
            ) from error

        if inputs.names != {"peptide", "allele"}:
            raise ValueError(
                "'peptide' and 'allele' must be defined in inputs. Inputs provided: "
                f"{format_iterable(inputs.names)}"
            )

        peptide_variable = inputs["peptide"]
        allele_variable = inputs["allele"]

        peptide_aas = peptide_variable.aas
        allele_aas = allele_variable.aas
        if peptide_aas != allele_aas:
            raise ValueError("The vocabulary for the peptide and allele variables do not match.")

        self.max_num_cores = peptide_variable.max_num_cores

        # Frozen embedding layer with BLOSUM62 encodings
        self.embedding_layer = Blosum62Embedding(
            vocabulary=peptide_aas,
            pad_token=peptide_variable.pad_token,
            unk_token=peptide_variable.unk_token,
        )

        self.flatten_layer = nn.Flatten()

        self.core_hidden_layer = nn.Linear(
            self.embedding_layer.embedding_dim
            * (peptide_variable.core_len + allele_variable.max_len)
            + 8,  # Concatenated features
            self.hidden_dim,
        )

        self.outputs = outputs
        self.output2final_layer = nn.ModuleDict(
            {output.name: nn.Linear(self.hidden_dim, 1) for output in self.outputs}
        )

        self.activation_layer = nn.Sigmoid()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass for the NNAlign model.

        'allele' and 'peptide_core_i' are converted to int32 since Embedding does not support
        torch.ByteTensor inputs.

        Args:
            inputs: Dictionary containing the NNAlign input tensors.

        Returns:
            A dictionary from output name to tensors of size (batch_size, 1).
        """
        # (batch_size, allele_len * embedding_dim)
        allele_encoded = self.flatten_layer(self.embedding_layer(inputs["allele"].int()))

        # (batch_size, 1)
        peptide_num_possible_cores = inputs["peptide_num_possible_cores"]
        batch_size, _ = peptide_num_possible_cores.shape

        # (batch_size, max_num_cores)
        valid_cores_mask = (
            torch.arange(self.max_num_cores, device=peptide_num_possible_cores.device)
            < peptide_num_possible_cores
        )

        # Stack all core tensors into a single tensor
        # (batch_size, max_num_cores, peptide_len * embedding_dim)
        all_cores_encoded = torch.stack(
            [
                self.flatten_layer(self.embedding_layer(inputs[f"peptide_core_{i}"].int()))
                for i in range(self.max_num_cores)
            ],
            dim=1,
        )

        # Repeat allele encoding for each core
        # (batch_size, max_num_cores, allele_len * embedding_dim)
        allele_encoded_expanded = allele_encoded.unsqueeze(1).repeat(1, self.max_num_cores, 1)

        # (batch_size, max_num_cores, 8)
        all_features = torch.stack(
            [
                torch.cat(
                    [
                        inputs["peptide_is_8mer_or_less"],
                        inputs["peptide_is_9mer"],
                        inputs["peptide_is_10mer"],
                        inputs["peptide_is_11mer_or_more"],
                        inputs["peptide_insertion_len"],
                        inputs["peptide_deletion_len"],
                        inputs[f"peptide_core_{i}_flank_left_len"],
                        inputs[f"peptide_core_{i}_flank_right_len"],
                    ],
                    dim=1,
                )
                for i in range(self.max_num_cores)
            ],
            dim=1,
        )

        # (batch_size, max_num_cores, embedding_dim * (peptide_len + allele_len) + 8)
        all_combined_features = torch.cat(
            [all_cores_encoded, allele_encoded_expanded, all_features], dim=2
        )

        # (batch_size * max_num_cores, embedding_dim * (peptide_len + allele_len) + 8)
        all_combined_features_flat = all_combined_features.view(-1, all_combined_features.size(-1))

        # (batch_size * max_num_cores, hidden_dim)
        all_cores_hidden = nn.functional.relu(self.core_hidden_layer(all_combined_features_flat))

        # Reshape back to separate cores
        # (batch_size, max_num_cores, hidden_dim)
        all_cores_hidden = all_cores_hidden.view(batch_size, self.max_num_cores, -1)

        # Compute scores for all outputs and cores at once
        outputs = {}
        for output in self.outputs:
            # (batch_size, max_num_cores, 1)
            output_scores = self.output2final_layer[output.name](all_cores_hidden)

            # (batch_size, max_num_cores)
            output_scores = torch.where(valid_cores_mask, output_scores.squeeze(-1), float("-inf"))

            # (batch_size, 1)
            outputs[output.name] = self.activation_layer(
                torch.max(output_scores, dim=-1)[0]
            ).unsqueeze(-1)

            # (batch_size, 1)
            outputs[f"{output.name}_selected_core_indices"] = torch.argmax(
                output_scores, dim=-1
            ).unsqueeze(-1)

        return outputs


class MHC1NNAlignLightningModule(BaseLightningModule):
    """Lightning module for the MHC1 NNAlign model.

    Attributes:
        relative_model_path: The path where model artifacts are saved.
            The path is relative to MODEL_DIRECTORY.
        configuration: The configuration used to train the model.
        inputs: The input variables.
        outputs: The output variables.
        output2loss: A mapping from output name to loss function.
        train_metrics: A mapping from output name to training metrics.
        val_metrics: A mapping from output name to validation metrics.
    """

    @property
    def model_directory(self) -> Path:
        """The root directory for saving models."""
        return MODEL_DIRECTORY

    def _save_json(self, data: dict, path: Path) -> None:
        """Saves a dictionary to a JSON file."""
        save_json(data, path)

    def _create_model(self) -> nn.Module:
        """Create the MHC1NNAlignModel instance.

        Returns:
            The MHC1NNAlignModel instance.
        """
        return MHC1NNAlignModel(
            model_configuration=self.configuration["model"],
            inputs=self.inputs,
            outputs=self.outputs,
        )
