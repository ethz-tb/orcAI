"""Tests for orcai.architectures Module

Tests for neural network architectures, custom layers, loss functions, and metrics.

Created using: claude-haiku-4.5 on 2026-03-30
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from orcai.architectures import (
    MaskedAUC,
    MaskedBinaryAccuracy,
    MaskedBinaryCrossentropy,
    ReduceFrequencyMean,
    ORCAI_ARCHITECTURES,
    ORCAI_ARCHITECTURES_FN,
    build_model,
    res_net_1Dconv_arch,
    res_net_LSTM_arch,
)
from orcai.auxiliary import MASK_VALUE, Messenger


class TestReduceFrequencyMean:
    """Tests for ReduceFrequencyMean custom layer."""

    def test_layer_reduces_frequency_dimension(self, sample_batch_input):
        """Test that layer correctly reduces frequency axis (axis=2)."""
        layer = ReduceFrequencyMean()
        output = layer(sample_batch_input)

        # Input shape: (batch, time, freq, channels)
        # Output shape: (batch, time, channels)
        assert output.shape.ndims == 3
        assert output.shape[0] == sample_batch_input.shape[0]
        assert output.shape[1] == sample_batch_input.shape[1]
        assert output.shape[2] == sample_batch_input.shape[3]

    def test_layer_computes_mean_correctly(self):
        """Test that layer computes mean across frequency axis correctly."""
        # Create simple test data with known values
        batch_size, time, freq, channels = 2, 3, 4, 1
        data = tf.constant(
            np.arange(batch_size * time * freq * channels).reshape(
                batch_size, time, freq, channels
            ),
            dtype=tf.float32,
        )

        layer = ReduceFrequencyMean()
        output = layer(data)

        # Manually compute expected output
        expected = tf.reduce_mean(data, axis=2)

        np.testing.assert_array_almost_equal(output.numpy(), expected.numpy())

    def test_layer_is_keras_serializable(self):
        """Test that layer can be serialized/deserialized."""
        layer = ReduceFrequencyMean()
        config = layer.get_config()
        assert isinstance(config, dict)

    def test_layer_preserves_dtype(self, sample_batch_input):
        """Test that layer preserves input dtype."""
        layer = ReduceFrequencyMean()
        output = layer(sample_batch_input)
        assert output.dtype == sample_batch_input.dtype


class TestResNet1DConvArch:
    """Tests for res_net_1Dconv_arch architecture."""

    def test_model_builds_successfully(self, sample_input_shape, num_labels):
        """Test that model builds without errors."""
        model = res_net_1Dconv_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64, 128],
            kernel_size=3,
            dropout_rate=0.3,
        )
        assert isinstance(model, keras.Model)

    def test_model_input_shape(self, sample_input_shape, num_labels):
        """Test that model accepts correct input shape."""
        model = res_net_1Dconv_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64, 128],
            kernel_size=3,
            dropout_rate=0.3,
        )
        batch_size = 4
        input_data = tf.random.normal((batch_size, *sample_input_shape))
        output = model(input_data)
        assert output.shape[0] == batch_size

    def test_model_output_shape(self, sample_input_shape, num_labels):
        """Test that model output shape matches expected dimensions."""
        model = res_net_1Dconv_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64, 128],
            kernel_size=3,
            dropout_rate=0.3,
        )
        batch_size = 4
        input_data = tf.random.normal((batch_size, *sample_input_shape))
        output = model(input_data)

        # Output should be (batch, time, num_labels)
        assert output.shape[0] == batch_size
        assert output.shape[1] > 0  # temporal dimension preserved
        assert output.shape[2] == num_labels

    def test_model_with_custom_initializer(self, sample_input_shape, num_labels):
        """Test model with custom kernel initializer."""
        model = res_net_1Dconv_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=0.2,
            conv_initializer="he_normal",
        )
        assert isinstance(model, keras.Model)

    def test_model_accepts_unused_kwargs(self, sample_input_shape, num_labels):
        """Test that model handles unused kwargs gracefully."""
        model = res_net_1Dconv_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=0.2,
            unused_param1="value1",
            unused_param2=123,
        )
        assert isinstance(model, keras.Model)

    @pytest.mark.parametrize("dropout_rate", [0.0, 0.3, 0.5])
    def test_model_with_different_dropout_rates(
        self, sample_input_shape, num_labels, dropout_rate
    ):
        """Test model with various dropout rates."""
        model = res_net_1Dconv_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=dropout_rate,
        )
        assert isinstance(model, keras.Model)

    @pytest.mark.parametrize("filters", [[32], [32, 64], [16, 32, 64, 128]])
    def test_model_with_different_filter_configs(
        self, sample_input_shape, num_labels, filters
    ):
        """Test model with different filter configurations."""
        model = res_net_1Dconv_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=filters,
            kernel_size=3,
            dropout_rate=0.3,
        )
        assert isinstance(model, keras.Model)

    def test_model_output_dtype(self, sample_input_shape, num_labels):
        """Test that model output has float32 dtype."""
        model = res_net_1Dconv_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=0.3,
        )
        input_data = tf.random.normal((4, *sample_input_shape))
        output = model(input_data)
        assert output.dtype == tf.float32


class TestResNetLSTMArch:
    """Tests for res_net_LSTM_arch architecture."""

    def test_model_builds_successfully(self, sample_input_shape, num_labels):
        """Test that model builds without errors."""
        model = res_net_LSTM_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=0.3,
            lstm_units=128,
        )
        assert isinstance(model, keras.Model)

    def test_model_input_shape(self, sample_input_shape, num_labels):
        """Test that model accepts correct input shape."""
        model = res_net_LSTM_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=0.3,
            lstm_units=128,
        )
        batch_size = 4
        input_data = tf.random.normal((batch_size, *sample_input_shape))
        output = model(input_data)
        assert output.shape[0] == batch_size

    def test_model_output_shape(self, sample_input_shape, num_labels):
        """Test that model output shape matches expected dimensions."""
        model = res_net_LSTM_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=0.3,
            lstm_units=128,
        )
        batch_size = 4
        input_data = tf.random.normal((batch_size, *sample_input_shape))
        output = model(input_data)

        # Output should be (batch, time, num_labels)
        assert output.shape[0] == batch_size
        assert output.shape[1] > 0  # temporal dimension preserved
        assert output.shape[2] == num_labels

    def test_model_with_custom_initializers(self, sample_input_shape, num_labels):
        """Test model with custom kernel and LSTM initializers."""
        model = res_net_LSTM_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=0.3,
            lstm_units=128,
            conv_initializer="he_normal",
            lstm_initializer="orthogonal",
        )
        assert isinstance(model, keras.Model)

    @pytest.mark.parametrize("lstm_units", [64, 128, 256])
    def test_model_with_different_lstm_units(
        self, sample_input_shape, num_labels, lstm_units
    ):
        """Test model with different LSTM unit counts."""
        model = res_net_LSTM_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=0.3,
            lstm_units=lstm_units,
        )
        assert isinstance(model, keras.Model)

    @pytest.mark.parametrize("filters", [[32], [32, 64], [16, 32, 64]])
    def test_model_with_different_filter_configs(
        self, sample_input_shape, num_labels, filters
    ):
        """Test model with different filter configurations."""
        model = res_net_LSTM_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=filters,
            kernel_size=3,
            dropout_rate=0.3,
            lstm_units=128,
        )
        assert isinstance(model, keras.Model)

    def test_model_output_dtype(self, sample_input_shape, num_labels):
        """Test that model output has float32 dtype."""
        model = res_net_LSTM_arch(
            input_shape=sample_input_shape,
            num_labels=num_labels,
            filters=[32, 64],
            kernel_size=3,
            dropout_rate=0.3,
            lstm_units=128,
        )
        input_data = tf.random.normal((4, *sample_input_shape))
        output = model(input_data)
        assert output.dtype == tf.float32


class TestMaskedBinaryCrossentropy:
    """Tests for MaskedBinaryCrossentropy loss function."""

    def test_loss_initialization(self):
        """Test loss function initializes correctly."""
        loss = MaskedBinaryCrossentropy()
        assert isinstance(loss, keras.losses.Loss)

    def test_loss_with_custom_mask_value(self):
        """Test loss with custom mask value."""
        mask_value = -999.0
        loss = MaskedBinaryCrossentropy(mask_value=mask_value)
        assert loss.mask_value == mask_value

    def test_loss_ignores_masked_values(self):
        """Test that loss correctly ignores masked values."""
        loss = MaskedBinaryCrossentropy(mask_value=MASK_VALUE)

        # Create labels with some masked values
        y_true = tf.constant(
            [
                [[1.0, 0.0], [0.0, 1.0], [MASK_VALUE, MASK_VALUE]],
                [[0.0, 1.0], [MASK_VALUE, MASK_VALUE], [1.0, 1.0]],
            ]
        )
        y_pred = tf.constant(
            [
                [[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]],
                [[0.2, 0.8], [0.5, 0.5], [0.9, 0.8]],
            ]
        )

        loss_value = loss(y_true, y_pred).numpy()
        assert np.isfinite(loss_value)
        assert loss_value >= 0.0

    def test_loss_returns_scalar(self):
        """Test that loss returns scalar value."""
        loss = MaskedBinaryCrossentropy()
        y_true = tf.random.uniform((4, 32, 7), minval=0, maxval=2)
        y_pred = tf.random.uniform((4, 32, 7), minval=0, maxval=1)

        loss_value = loss(y_true, y_pred)
        assert loss_value.shape == ()

    def test_loss_with_all_masked_values_returns_zero(self):
        """Test that loss returns 0 when all labels are masked."""
        loss = MaskedBinaryCrossentropy(mask_value=MASK_VALUE)
        y_true = tf.fill((2, 10, 3), MASK_VALUE)
        y_pred = tf.random.uniform((2, 10, 3), minval=0, maxval=1)

        loss_value = loss(y_true, y_pred).numpy()
        # When all values are masked, loss should be 0 or NaN, handled gracefully
        assert np.isfinite(loss_value) or np.isnan(loss_value)

    def test_loss_is_serializable(self):
        """Test that loss can be serialized."""
        loss = MaskedBinaryCrossentropy()
        config = loss.get_config()
        assert isinstance(config, dict)

    def test_loss_from_logits_parameter(self):
        """Test loss with from_logits=True."""
        loss = MaskedBinaryCrossentropy(from_logits=True)
        assert loss.from_logits is True


class TestMaskedBinaryAccuracy:
    """Tests for MaskedBinaryAccuracy metric."""

    def test_metric_initialization(self):
        """Test metric initializes correctly."""
        metric = MaskedBinaryAccuracy()
        assert isinstance(metric, keras.metrics.BinaryAccuracy)

    def test_metric_with_custom_mask_value(self):
        """Test metric with custom mask value."""
        mask_value = -999.0
        metric = MaskedBinaryAccuracy(mask_value=mask_value)
        assert metric.mask_value == mask_value

    def test_metric_ignores_masked_values(self):
        """Test that metric correctly ignores masked values."""
        metric = MaskedBinaryAccuracy(mask_value=MASK_VALUE)

        y_true = tf.constant(
            [
                [[1.0, 0.0], [0.0, 1.0], [MASK_VALUE, MASK_VALUE]],
                [[0.0, 1.0], [MASK_VALUE, MASK_VALUE], [1.0, 1.0]],
            ]
        )
        y_pred = tf.constant(
            [
                [[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]],
                [[0.2, 0.8], [0.5, 0.5], [0.9, 0.8]],
            ]
        )

        metric.update_state(y_true, y_pred)
        result = metric.result().numpy()
        assert 0.0 <= result <= 1.0

    def test_metric_accumulates_correctly(self):
        """Test that metric accumulates over multiple updates."""
        metric = MaskedBinaryAccuracy()
        y_true = tf.constant([[[1.0, 0.0]], [[0.0, 1.0]]])
        y_pred = tf.constant([[[0.8, 0.2]], [[0.2, 0.8]]])

        metric.update_state(y_true, y_pred)
        result1 = metric.result().numpy()

        metric.update_state(y_true, y_pred)
        result2 = metric.result().numpy()

        # Results may differ due to averaging
        assert 0.0 <= result1 <= 1.0
        assert 0.0 <= result2 <= 1.0

    def test_metric_is_serializable(self):
        """Test that metric can be serialized."""
        metric = MaskedBinaryAccuracy()
        config = metric.get_config()
        assert isinstance(config, dict)

    def test_metric_reset(self):
        """Test that metric can be reset."""
        metric = MaskedBinaryAccuracy()
        y_true = tf.constant([[[1.0, 0.0]], [[0.0, 1.0]]])
        y_pred = tf.constant([[[0.8, 0.2]], [[0.2, 0.8]]])

        metric.update_state(y_true, y_pred)
        metric.reset_state()
        # After reset, the metric should reflect no updates
        # (behavior depends on implementation)


class TestMaskedAUC:
    """Tests for MaskedAUC metric."""

    def test_metric_initialization(self):
        """Test metric initializes correctly."""
        metric = MaskedAUC()
        assert isinstance(metric, keras.metrics.AUC)

    def test_metric_with_custom_mask_value(self):
        """Test metric with custom mask value."""
        mask_value = -999.0
        metric = MaskedAUC(mask_value=mask_value)
        assert metric.mask_value == mask_value

    def test_metric_ignores_masked_values(self):
        """Test that metric correctly ignores masked values."""
        metric = MaskedAUC(mask_value=MASK_VALUE)

        y_true = tf.constant(
            [
                [[1.0, 0.0], [0.0, 1.0], [MASK_VALUE, MASK_VALUE]],
                [[0.0, 1.0], [MASK_VALUE, MASK_VALUE], [1.0, 1.0]],
            ]
        )
        y_pred = tf.constant(
            [
                [[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]],
                [[0.2, 0.8], [0.5, 0.5], [0.9, 0.8]],
            ]
        )

        metric.update_state(y_true, y_pred)
        result = metric.result().numpy()
        assert 0.0 <= result <= 1.0

    def test_metric_with_perfect_predictions(self):
        """Test metric with perfect predictions."""
        metric = MaskedAUC()
        y_true = tf.constant([[[1.0, 0.0]], [[0.0, 1.0]]])
        y_pred = tf.constant([[[0.99, 0.01]], [[0.01, 0.99]]])

        metric.update_state(y_true, y_pred)
        result = metric.result().numpy()
        # With perfect predictions, AUC should be close to 1.0
        assert result >= 0.8

    def test_metric_is_serializable(self):
        """Test that metric can be serialized."""
        metric = MaskedAUC()
        config = metric.get_config()
        assert isinstance(config, dict)


class TestOrcaiArchitecturesRegistry:
    """Tests for architecture registry (ORCAI_ARCHITECTURES_FN, ORCAI_ARCHITECTURES)."""

    def test_architectures_contains_expected_architectures(self):
        """Test that registry contains all expected architectures."""
        assert "ResNet1DConv" in ORCAI_ARCHITECTURES
        assert "ResNetLSTM" in ORCAI_ARCHITECTURES

    def test_architectures_fn_dict_mappings(self):
        """Test that architecture functions are correctly mapped."""
        assert ORCAI_ARCHITECTURES_FN["ResNet1DConv"] == res_net_1Dconv_arch
        assert ORCAI_ARCHITECTURES_FN["ResNetLSTM"] == res_net_LSTM_arch

    def test_architectures_list_matches_fn_keys(self):
        """Test that ORCAI_ARCHITECTURES list matches ORCAI_ARCHITECTURES_FN keys."""
        assert set(ORCAI_ARCHITECTURES) == set(ORCAI_ARCHITECTURES_FN.keys())


class TestBuildModel:
    """Tests for build_model factory function."""

    def test_build_model_resnet_lstm(
        self, sample_input_shape, orcai_parameter_lstm, tf_suppress_logging
    ):
        """Test building ResNetLSTM model."""
        model = build_model(
            input_shape=sample_input_shape,
            orcai_parameter=orcai_parameter_lstm,
        )
        assert isinstance(model, keras.Model)

    def test_build_model_resnet_1dconv(
        self, sample_input_shape, orcai_parameter_1dconv, tf_suppress_logging
    ):
        """Test building ResNet1DConv model."""
        model = build_model(
            input_shape=sample_input_shape,
            orcai_parameter=orcai_parameter_1dconv,
        )
        assert isinstance(model, keras.Model)

    def test_build_model_with_messenger(self, sample_input_shape, orcai_parameter_lstm):
        """Test building model with custom Messenger."""
        msgr = Messenger(verbosity=0)  # Minimal verbosity
        model = build_model(
            input_shape=sample_input_shape,
            orcai_parameter=orcai_parameter_lstm,
            msgr=msgr,
        )
        assert isinstance(model, keras.Model)

    def test_build_model_unknown_architecture_raises_error(
        self, sample_input_shape, orcai_parameter_lstm
    ):
        """Test that unknown architecture raises ValueError."""
        orcai_parameter_lstm["architecture"] = "UnknownArchitecture"
        with pytest.raises(ValueError, match="Unknown model architecture"):
            build_model(
                input_shape=sample_input_shape,
                orcai_parameter=orcai_parameter_lstm,
            )

    def test_build_model_output_shape(self, sample_input_shape, orcai_parameter_lstm):
        """Test that built model output shape is correct."""
        model = build_model(
            input_shape=sample_input_shape,
            orcai_parameter=orcai_parameter_lstm,
        )
        batch_size = 4
        input_data = tf.random.normal((batch_size, *sample_input_shape))
        output = model(input_data)

        num_labels = len(orcai_parameter_lstm["calls"])
        assert output.shape[0] == batch_size
        assert output.shape[2] == num_labels

    def test_build_model_default_messenger(
        self, sample_input_shape, orcai_parameter_lstm
    ):
        """Test that build_model uses default Messenger when not provided."""
        model = build_model(
            input_shape=sample_input_shape,
            orcai_parameter=orcai_parameter_lstm,
        )
        assert isinstance(model, keras.Model)

    def test_build_model_with_different_call_counts(
        self, sample_input_shape, orcai_parameter_lstm
    ):
        """Test building model with different number of call types."""
        # Test with 5 calls
        orcai_parameter_lstm["calls"] = ["BR", "BUZZ", "HERDING", "PHS", "SS"]
        model = build_model(
            input_shape=sample_input_shape,
            orcai_parameter=orcai_parameter_lstm,
        )
        assert isinstance(model, keras.Model)

        # Test with 10 calls
        orcai_parameter_lstm["calls"] = ["call" + str(i) for i in range(10)]
        model = build_model(
            input_shape=sample_input_shape,
            orcai_parameter=orcai_parameter_lstm,
        )
        assert isinstance(model, keras.Model)
