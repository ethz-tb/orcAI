"""Tests for train module.

Tests for the _count_params helper function.

Created using: claude-sonnet-4-6 on 2026-03-31
"""

import keras

from orcai.train import _count_params


class TestCountParams:
    """Tests for _count_params."""

    def test_dense_layer_count(self):
        """Dense(10, input_shape=(5,)) has 5*10 weights + 10 biases = 60 params."""
        model = keras.Sequential([keras.layers.Dense(10, input_shape=(5,))])
        count = _count_params(model.trainable_weights)
        assert count == 60

    def test_zero_for_empty_list(self):
        """Empty weight list returns 0."""
        assert _count_params([]) == 0

    def test_non_trainable_weights_excluded(self):
        """Only trainable weights are counted when passing trainable_weights."""
        model = keras.Sequential([keras.layers.Dense(4, input_shape=(3,))])
        trainable = _count_params(model.trainable_weights)
        # 3*4 + 4 = 16
        assert trainable == 16

    def test_multi_layer_model(self):
        """Counts params across multiple layers."""
        model = keras.Sequential(
            [
                keras.layers.Dense(8, input_shape=(4,)),  # 4*8+8 = 40
                keras.layers.Dense(2),  # 8*2+2 = 18
            ]
        )
        count = _count_params(model.trainable_weights)
        assert count == 58

    def test_returns_int_or_numpy_scalar(self):
        """Return type is an integer-like scalar."""
        model = keras.Sequential([keras.layers.Dense(3, input_shape=(2,))])
        count = _count_params(model.trainable_weights)
        assert int(count) == count
