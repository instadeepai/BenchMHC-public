"""Utils functions linked to predictions post-processing."""

from collections import defaultdict

import numpy as np
import torch


def process_outputs(batched_predictions: list[dict[str, torch.Tensor]]) -> dict[str, np.ndarray]:
    """Process output preditions of the lightning Trainer.

    >>> batch = {
    ...     'hit': torch.tensor([[0.1122], [0.1298]]),
    ...     'binding_affinity': torch.tensor([[0.4877], [0.4902]])
    ... }
    >>> result = process_outputs([batch,batch])
    >>> np.allclose(result['hit'], np.array([0.1122, 0.1298,0.1122, 0.1298]))
    True
    >>> np.allclose(result['binding_affinity'], np.array([0.4877, 0.4902,0.4877, 0.4902]))
    True

    Args:
        batched_predictions: The predictions on a dataset.

    Returns:
        The un-nested predictions.
    """
    output_name2predictions = defaultdict(list)
    for output_name2batch_predictions in batched_predictions:
        for output_name, batch_predictions in output_name2batch_predictions.items():
            output_name2predictions[output_name].append(batch_predictions.numpy().squeeze())

    return {
        output_name: np.concatenate(predictions)
        for output_name, predictions in output_name2predictions.items()
    }
