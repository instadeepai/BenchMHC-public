"""Module to define custom layers."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from Bio.Align import substitution_matrices

from bench_mhc.utils.format import format_iterable


class Blosum62Embedding(torch.nn.Embedding):
    """Class to implement a frozen embedding layer initialized with BLOSUM62 encodings.

    Attributes:
        vocabulary: Vocabulary of the embedding.
        pad_token: Padding token.
        unk_token: Unknown token.
        num_embeddings: Number of embeddings.
        embedding_dim: Embedding dimension.
    """

    def __init__(
        self,
        vocabulary: Iterable[str],
        pad_token: str | None,
        unk_token: str | None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Blosum62Embedding layer with a given vocabulary.

        We consider that the PAD and UNK tokens are safe-inserted
        (i.e. indices 0 and 1 of the 'aa2idx' mapping).

        Args:
            vocabulary: Vocabulary of the embedding.
            pad_token: Padding token.
            unk_token: Unknown token.
            kwargs: Keyword arguments to pass to the parent torch.nn.Embedding.

        Raises:
            ValueError: If unk_token or pad_token is defined and not in the vocabulary.
        """
        self.vocabulary = list(vocabulary)
        self.pad_token = pad_token
        self.unk_token = unk_token

        if self.pad_token is not None and self.pad_token not in self.vocabulary:
            raise ValueError(
                f"'pad_token={self.pad_token}' is not in the vocabulary provided to the "
                f"Blosum62Embedding layer. Provided vocabulary: {format_iterable(self.vocabulary)}."
            )

        if self.unk_token is not None and self.unk_token not in self.vocabulary:
            raise ValueError(
                f"'unk_token={self.unk_token}' is not in the vocabulary provided to the "
                f"Blosum62Embedding layer. Provided vocabulary: {format_iterable(self.vocabulary)}."
            )

        num_embeddings = len(self.vocabulary)
        embedding_dim = (
            len(self.vocabulary) - int(self.unk_token is not None) - int(self.pad_token is not None)
        )

        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim, **kwargs)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights with the BLOSUM62 encoding for each token in the vocabulary.

        The weights are a tensor with the i-th row being the BLOSUM62 encoding corresponding to
        the i-th element of the vocabulary of shape (vocabulary_len, vocabulary_len - 2),
        PAD and UNK tokens being encoded with Os.
        """
        vocabulary_wo_pad_and_unk = set(self.vocabulary) - {self.pad_token, self.unk_token}
        blosum62_mat = substitution_matrices.load("BLOSUM62")  # type: ignore
        blosum62_scores = np.zeros((self.num_embeddings, self.embedding_dim))

        for i, token in enumerate(self.vocabulary):
            # For PAD and UNK, we encode with 0s
            if token in {self.pad_token, self.unk_token}:
                continue
            else:
                blosum62_scores[i] = np.array(
                    [
                        blosum62_mat[(token, aa)]
                        for aa in blosum62_mat.alphabet
                        if aa in vocabulary_wo_pad_and_unk
                    ]
                )

        # Set the frozen weights
        with torch.no_grad():
            self.weight.copy_(torch.tensor(blosum62_scores, dtype=self.weight.dtype))
            self.weight.requires_grad = False
