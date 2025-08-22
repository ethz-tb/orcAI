import pytest
import numpy as np
import keras
import tensorflow as tf
from orcai.architectures import (
    res_net_1Dconv_arch,
    res_net_LSTM_arch,
    MaskedBinaryCrossentropy,
    MaskedBinaryAccuracy,
    MaskedAUC,
    build_model,
)

# Dummy parameters for testing
input_shape = [736, 171, 1]
num_labels = 7
filters = [30, 40, 50, 60]
kernel_size = 5
dropout_rate = 0.3
lstm_units = 128


@pytest.mark.parametrize("arch_fn", [res_net_1Dconv_arch, res_net_LSTM_arch])
def test_model_build_and_forward(arch_fn):
    if arch_fn is res_net_LSTM_arch:
        model = arch_fn(
            input_shape, num_labels, filters, kernel_size, dropout_rate, lstm_units
        )
    else:
        model = arch_fn(input_shape, num_labels, filters, kernel_size, dropout_rate)
    x = np.random.rand(2, *input_shape).astype(np.float32)
    y = model(x)
    assert y.shape[0] == 2
    assert y.shape[-1] == num_labels


def test_build_model():
    orcai_parameter = {
        "name": "test_model",
        "architecture": "ResNet1DConv",
        "calls": ["A", "B", "C"],
        "model": {
            "filters": filters,
            "kernel_size": kernel_size,
            "dropout_rate": dropout_rate,
        },
    }
    model = build_model(input_shape, orcai_parameter)
    assert isinstance(model, keras.Model)


def test_masked_binary_crossentropy():
    loss = MaskedBinaryCrossentropy(mask_value=-1)
    y_true = tf.constant([[1, 0, -1], [0, 1, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1, 0.5], [0.2, 0.8, 0.7]], dtype=tf.float32)
    val = loss(y_true, y_pred).numpy()
    assert np.isfinite(val)


def test_masked_binary_accuracy():
    metric = MaskedBinaryAccuracy(mask_value=-1)
    y_true = tf.constant([[1, 0, -1], [0, 1, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1, 0.5], [0.2, 0.8, 0.7]], dtype=tf.float32)
    metric.update_state(y_true, y_pred)
    val = metric.result().numpy()
    assert 0.0 <= val <= 1.0


def test_masked_auc():
    metric = MaskedAUC(mask_value=-1)
    y_true = tf.constant([[1, 0, -1], [0, 1, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1, 0.5], [0.2, 0.8, 0.7]], dtype=tf.float32)
    metric.update_state(y_true, y_pred)
    val = metric.result().numpy()
    assert 0.0 <= val <= 1.0
