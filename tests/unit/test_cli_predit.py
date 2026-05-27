"""Unit tests for the command lines in bench_mhc/cli/predict.py."""

import numpy as np

from bench_mhc.cli.predict import postprocess_predictions


def test_postprocess_predictions_multiple_submodels() -> None:
    """Test that the postprocess_predictions function works correctly.

    We test the functions with 2 submodels and 3 data points.
    """
    output_name2predictions = {
        "dummy_hit_selected_core_indices": np.array(
            # First line of 0 comes from the initialisation and will be removed in
            # `postprocess_predictions`
            [
                [0, 0, 0],
                # First submodel 3 data points
                [0, 2, 2],
                # Second submodel 3 data points
                [0, 0, 2],
            ]
        ),
        "dummy_binding_affinity_selected_core_indices": np.array(
            [
                [0, 0, 0],
                [0, 2, 2],
                [0, 0, 2],
            ]
        ),
        # Sumed probabilities of the 2 submodels on the 3 data points
        "dummy_hit": np.array([1, 0, 0]),
        "dummy_binding_affinity": np.array([1, 0, 0]),
    }
    # Count = number of submodels per output
    output_name2count = {
        "dummy_hit_selected_core_indices": 2,
        "dummy_binding_affinity_selected_core_indices": 2,
        "dummy_hit": 2,
        "dummy_binding_affinity": 2,
    }

    output_name2predictions_postprocessed = postprocess_predictions(
        output_name2predictions, output_name2count
    )

    expected_output_name2predictions_postprocessed = {
        "dummy_hit_selected_core_indices": np.array([[0, 0], [2, 0], [2, 2]]).astype(str),
        "dummy_binding_affinity_selected_core_indices": np.array([[0, 0], [2, 0], [2, 2]]).astype(
            str
        ),
        "dummy_hit": np.array([1 / 2, 0, 0]),
        "dummy_binding_affinity": np.array([1 / 2, 0, 0]),
        "dummy_hit_majority_core_index": np.array([0, 0, 2]),
        "dummy_binding_affinity_majority_core_index": np.array([0, 0, 2]),
    }

    for output_name, predictions in output_name2predictions_postprocessed.items():
        assert (predictions == expected_output_name2predictions_postprocessed[output_name]).all()


def test_postprocess_predictions_single_model() -> None:
    """Test that the postprocess_predictions function works correctly.

    We test the functions with 1 submodel on 3 data points.
    """
    output_name2predictions = {
        "dummy_hit": np.array([1, 0, 0]),
        "dummy_binding_affinity": np.array([1, 0, 0]),
        "dummy_hit_selected_core_indices": np.array([[0, 0, 0], [1, 2, 3]]),
        "dummy_binding_affinity_selected_core_indices": np.array([[0, 0, 0], [1, 2, 3]]),
    }
    output_name2count = {
        "dummy_hit": 1,
        "dummy_binding_affinity": 1,
        "dummy_hit_selected_core_indices": 1,
        "dummy_binding_affinity_selected_core_indices": 1,
    }

    expected_output_name2predictions_postprocessed = {
        "dummy_hit": np.array([1, 0, 0]),
        "dummy_binding_affinity": np.array([1, 0, 0]),
        "dummy_hit_selected_core_indices": np.array([[1], [2], [3]]).astype(str),
        "dummy_binding_affinity_selected_core_indices": np.array([[1], [2], [3]]).astype(str),
        "dummy_hit_majority_core_index": np.array([1, 2, 3]),
        "dummy_binding_affinity_majority_core_index": np.array([1, 2, 3]),
    }

    output_name2predictions_postprocessed = postprocess_predictions(
        output_name2predictions, output_name2count
    )

    for output_name, predictions in output_name2predictions_postprocessed.items():
        assert (predictions == expected_output_name2predictions_postprocessed[output_name]).all()
