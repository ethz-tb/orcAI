"""Pytest Configuration and Shared Fixtures

Shared pytest fixtures for OrcAI tests including sample inputs, labels, and model parameters.

Created using: claude-haiku-4.5 on 2026-03-31
"""

import pytest
import tensorflow as tf


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
