"""Pytest Configuration and Shared Fixtures

Shared pytest fixtures for OrcAI tests including sample inputs, labels, and model parameters.

Created using: claude-haiku-4.5 on 2026-03-31
"""

import io
import json

import pytest
import tensorflow as tf

from orcai.auxiliary import Messenger


@pytest.fixture(scope="session")
def tf_suppress_logging():
    """Suppress TensorFlow logging during tests."""
    tf.get_logger().setLevel(40)


@pytest.fixture(scope="function")
def sample_input_shape():
    """Standard input shape for spectrogram data: (time, freq, channels)."""
    return (128, 64, 1)


@pytest.fixture(scope="function")
def sample_batch_input(sample_input_shape):
    """Generate sample batch input for testing.

    Shape: (batch_size, time, freq, channels)
    """
    batch_size = 4
    shape = (batch_size, *sample_input_shape)
    return tf.random.normal(shape)


@pytest.fixture(scope="function")
def num_labels():
    """Standard number of call type labels."""
    return 7


@pytest.fixture(scope="function")
def sample_model_params(num_labels):
    """Standard model hyperparameters for ResNet architectures."""
    return {
        "filters": [32, 64, 128],
        "kernel_size": 3,
        "dropout_rate": 0.3,
        "lstm_units": 128,
    }


@pytest.fixture(scope="function")
def orcai_parameter_lstm(num_labels, sample_model_params, sample_input_shape):
    """OrcAI parameter dict for ResNetLSTM architecture."""
    return {
        "name": "test-model-lstm",
        "architecture": "ResNetLSTM",
        "calls": ["BR", "BUZZ", "HERDING", "PHS", "SS", "TAILSLAP", "WHISTLE"],
        "model": sample_model_params,
    }


@pytest.fixture(scope="function")
def orcai_parameter_1dconv(num_labels, sample_model_params, sample_input_shape):
    """OrcAI parameter dict for ResNet1DConv architecture."""
    return {
        "name": "test-model-1dconv",
        "architecture": "ResNet1DConv",
        "calls": ["BR", "BUZZ", "HERDING", "PHS", "SS", "TAILSLAP", "WHISTLE"],
        "model": sample_model_params,
    }


@pytest.fixture(scope="function")
def sample_labels(num_labels):
    """Generate sample binary labels.

    Shape: (batch_size, time, num_labels)
    Values: 0 or 1 only (no masked values).
    """
    batch_size = 4
    time_steps = 64
    return tf.random.uniform(
        (batch_size, time_steps, num_labels), minval=0, maxval=2, dtype=tf.float32
    )


@pytest.fixture(scope="function")
def sample_labels_with_mask(num_labels):
    """Generate sample labels with masked values (-1.0).

    Shape: (batch_size, time, num_labels)
    Values: -1.0 (masked), 0, or 1 (valid labels).
    """
    batch_size = 4
    time_steps = 64
    labels = tf.random.uniform(
        (batch_size, time_steps, num_labels), minval=0, maxval=2, dtype=tf.float32
    )
    # Add some masked values
    mask_indices = tf.random.uniform((batch_size, time_steps, 1), 0, 1) > 0.8
    mask_indices = tf.tile(mask_indices, (1, 1, num_labels))
    labels = tf.where(mask_indices, tf.constant(-1.0), labels)
    return labels


@pytest.fixture(scope="function")
def test_messenger():
    """Create a test Messenger with string output."""
    output = io.StringIO()
    return Messenger(verbosity=2, file=output), output


@pytest.fixture(scope="function")
def hps_parameter_simple():
    """Simple hyperparameter search parameter for testing."""
    return {
        "filters": {"f1": [32, 64], "f2": [64, 128]},
        "kernel_size": [3, 5],
        "dropout_rate": [0.2, 0.3],
        "batch_size": [16, 32],
        "tuner": {"max_epochs": 2, "early_stopping_patience": 1},
    }


@pytest.fixture(scope="function")
def hps_parameter_simple_with_lstm(hps_parameter_simple):
    """Hyperparameter search parameter with LSTM units for testing."""
    param = hps_parameter_simple.copy()
    param["lstm_units"] = [128, 256]
    return param


@pytest.fixture(scope="function")
def hps_parameter_minimal():
    """Minimal hyperparameter search parameter for testing."""
    return {
        "filters": {"f1": [32]},
        "kernel_size": [3],
        "dropout_rate": [0.3],
        "batch_size": [32],
        "tuner": {"max_epochs": 1, "early_stopping_patience": 1},
    }


@pytest.fixture(scope="function")
def orcai_parameter_hpsearch():
    """OrcAI parameter for hyperparameter search testing."""
    return {
        "name": "test",
        "seed": 42,
        "architecture": "ResNet1DConv",
        "calls": ["BR", "BUZZ"],
        "model": {
            "filters": [32],
            "kernel_size": 3,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "monitor": "val_loss",
            "batch_size": 32,
        },
    }


@pytest.fixture(scope="function")
def orcai_parameter_hpsearch_lstm():
    """OrcAI parameter for LSTM hyperparameter search testing."""
    return {
        "name": "test",
        "seed": 42,
        "architecture": "ResNetLSTM",
        "calls": ["BR"],
        "model": {
            "filters": [32],
            "kernel_size": 3,
            "dropout_rate": 0.3,
            "lstm_units": 128,
            "learning_rate": 0.001,
            "monitor": "val_loss",
            "batch_size": 32,
        },
    }


@pytest.fixture(scope="function")
def dataset_shapes_fixture(sample_input_shape):
    """Dataset shapes for hyperparameter search testing."""
    return {"spectrogram": list(sample_input_shape)}


@pytest.fixture(scope="function")
def data_dir_with_shapes(tmp_path, dataset_shapes_fixture):
    """Create a temporary data directory with dataset_shapes.json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with open(data_dir / "dataset_shapes.json", "w") as f:
        json.dump(dataset_shapes_fixture, f)
    return data_dir


@pytest.fixture(scope="function")
def recordings_structure(tmp_path):
    """Create a temporary recordings directory structure with test files."""
    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir()
    (recording_dir / "test1.wav").touch()
    (recording_dir / "test2.wav").touch()

    subdir = recording_dir / "subdir"
    subdir.mkdir()
    (subdir / "test3.wav").touch()

    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    (annotation_dir / "test1.txt").touch()
    (annotation_dir / "test2.txt").touch()

    return {
        "recording_dir": recording_dir,
        "annotation_dir": annotation_dir,
        "tmp_path": tmp_path,
    }
