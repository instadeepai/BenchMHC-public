"""Unit tests related to bench_mhc/custom_objects/layers.py."""

import numpy as np
import pytest
import torch
from Bio.Align import substitution_matrices
from torch.nn import Embedding

from bench_mhc.constants import PAD_TOKEN
from bench_mhc.custom_objects.layers import Blosum62Embedding
from bench_mhc.utils.format import format_iterable
from bench_mhc.variables import AASeqVariable


class TestBlosum62Embedding:
    """Test class for Blosum62Embedding layer."""

    @pytest.mark.parametrize("pad_token", [None, PAD_TOKEN])
    @pytest.mark.parametrize("unk_token", [None, "X"])
    def test_blosum62_embedding(self, pad_token: str | None, unk_token: str | None) -> None:
        """Test the Blosum62Embedding custom layer."""
        vocabulary = ["T", "E", "S"]

        if pad_token is not None:
            match_msg = (
                f"'pad_token={pad_token}' is not in the vocabulary provided to the "
                f"Blosum62Embedding layer. Provided vocabulary: {format_iterable(vocabulary)}."
            )
            with pytest.raises(ValueError, match=match_msg):
                _ = Blosum62Embedding(
                    vocabulary=vocabulary,
                    pad_token=pad_token,
                    unk_token=unk_token,
                )

            AASeqVariable._safe_insert_token_in_vocabulary(vocabulary, pad_token)

        if unk_token is not None:
            match_msg = (
                f"'unk_token={unk_token}' is not in the vocabulary provided to the "
                f"Blosum62Embedding layer. Provided vocabulary: {format_iterable(vocabulary)}."
            )
            with pytest.raises(ValueError, match=match_msg):
                _ = Blosum62Embedding(
                    vocabulary=vocabulary,
                    pad_token=pad_token,
                    unk_token=unk_token,
                )

            AASeqVariable._safe_insert_token_in_vocabulary(vocabulary, unk_token)

        blosum62_embedding = Blosum62Embedding(
            vocabulary=vocabulary,
            pad_token=pad_token,
            unk_token=unk_token,
        )

        assert isinstance(blosum62_embedding, Embedding)
        assert blosum62_embedding.vocabulary == vocabulary
        assert blosum62_embedding.pad_token == pad_token
        assert blosum62_embedding.unk_token == unk_token
        assert blosum62_embedding.num_embeddings == 3 + int(pad_token is not None) + int(
            unk_token is not None
        )
        assert blosum62_embedding.embedding_dim == 3
        assert not blosum62_embedding.weight.requires_grad

        blosum62_mat = substitution_matrices.load("BLOSUM62")  # type: ignore
        if pad_token is not None:
            torch.testing.assert_close(
                blosum62_embedding.weight[0],
                torch.from_numpy(np.array([0, 0, 0], dtype=np.float32)),
            )

        if unk_token is not None:
            torch.testing.assert_close(
                blosum62_embedding.weight[1 if pad_token is not None else 0],
                torch.from_numpy(np.array([0, 0, 0], dtype=np.float32)),
            )

        for i, token in enumerate(
            ["T", "E", "S"], start=int(pad_token is not None) + int(unk_token is not None)
        ):
            torch.testing.assert_close(
                blosum62_embedding.weight[i],
                torch.from_numpy(
                    np.array(
                        [
                            blosum62_mat[(token, aa)]
                            for aa in blosum62_mat.alphabet
                            if aa in ["T", "E", "S"]
                        ],
                        dtype=np.float32,
                    )
                ),
            )
