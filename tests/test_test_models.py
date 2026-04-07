"""Tests for test_models module.

Tests for confusion table, misclassification table, and batch-stacking helpers.

Created using: claude-sonnet-4-6 on 2026-03-31
"""

import numpy as np
import pytest

from orcai.auxiliary import MASK_VALUE
from orcai.test_models import (
    _compute_misclassification_table,
    _get_mask_for_rows_with_atmost_one_1,
    _stack_batch,
    compute_confusion_table,
    compute_misclassification_tables,
)


# ---------------------------------------------------------------------------
# _stack_batch
# ---------------------------------------------------------------------------


class TestStackBatch:
    """Tests for _stack_batch."""

    def test_stacks_matrices(self):
        """Vertically stacks a list of matrices."""
        batch = [np.ones((2, 3)), np.zeros((2, 3))]
        result = _stack_batch(batch)
        assert result.shape == (4, 3)

    def test_converts_to_int(self):
        """Output dtype is int."""
        batch = [np.array([[0.9, 0.1]])]
        result = _stack_batch(batch)
        assert result.dtype == int

    def test_single_element(self):
        """Works with a single-element batch."""
        batch = [np.eye(3)]
        result = _stack_batch(batch)
        assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# _get_mask_for_rows_with_atmost_one_1
# ---------------------------------------------------------------------------


class TestGetMaskForRowsWithAtmostOne1:
    """Tests for _get_mask_for_rows_with_atmost_one_1."""

    def test_all_zeros_row_included(self):
        """Row of all zeros (0 ones) is included (True)."""
        m = np.array([[0, 0, 0]])
        assert _get_mask_for_rows_with_atmost_one_1(m)[0]

    def test_single_one_included(self):
        """Row with exactly one 1 is included."""
        m = np.array([[0, 1, 0]])
        assert _get_mask_for_rows_with_atmost_one_1(m)[0]

    def test_two_ones_excluded(self):
        """Row with two 1s is excluded (False)."""
        m = np.array([[1, 1, 0]])
        assert not _get_mask_for_rows_with_atmost_one_1(m)[0]

    def test_mixed_rows(self):
        """Returns correct mask for a mixed matrix."""
        m = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
        mask = _get_mask_for_rows_with_atmost_one_1(m)
        assert list(mask) == [True, False, True]


# ---------------------------------------------------------------------------
# compute_confusion_table
# ---------------------------------------------------------------------------


class TestComputeConfusionTable:
    """Tests for compute_confusion_table."""

    def _perfect_batch(self, n_labels: int = 2, batch: int = 4, time: int = 8) -> tuple:
        """y_true == y_pred → all TP or TN."""
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=(batch, time, n_labels)).astype(float)
        return y, y.copy()

    def _zero_batch(self, n_labels: int = 2, batch: int = 4, time: int = 8) -> tuple:
        """y_true all-zero, y_pred all-zero."""
        y = np.zeros((batch, time, n_labels))
        return y, y.copy()

    def test_returns_dataframe(self):
        """Returns a pandas DataFrame."""
        import pandas as pd

        y_true, y_pred = self._perfect_batch()
        result = compute_confusion_table(y_true, y_pred, ["A", "B"])
        assert isinstance(result, pd.DataFrame)

    def test_column_names(self):
        """Output has TP, FN, FP, TN, PR, RE, F1, Total columns."""
        y_true, y_pred = self._perfect_batch()
        result = compute_confusion_table(y_true, y_pred, ["A", "B"])
        for col in ["TP", "FN", "FP", "TN", "PR", "RE", "F1", "Total"]:
            assert col in result.columns

    def test_rows_are_label_names(self):
        """One row per label."""
        y_true, y_pred = self._perfect_batch(n_labels=3)
        result = compute_confusion_table(y_true, y_pred, ["A", "B", "C"])
        assert set(result.index) == {"A", "B", "C"}

    def test_perfect_predictions_high_f1(self):
        """Perfect predictions yield F1 = 1.0 (or NaN if no positive labels)."""
        y_true, y_pred = self._perfect_batch()
        result = compute_confusion_table(y_true, y_pred, ["A", "B"])
        for f1 in result["F1"]:
            assert np.isnan(f1) or f1 == pytest.approx(1.0, abs=0.01)

    def test_mask_value_excluded(self):
        """Masked values (MASK_VALUE) are excluded from confusion computation."""
        y_true = np.zeros((2, 4, 1))
        y_pred = np.zeros((2, 4, 1))
        # Mask half the values
        y_true[0, :2, 0] = MASK_VALUE
        result = compute_confusion_table(y_true, y_pred, ["A"])
        # 2 batches × 4 time steps × 1 label = 8 total, 2 masked → 6 unmasked
        assert result.loc["A", "Total"] == 6

    def test_tp_fp_fn_tn_sum_to_one(self):
        """TP + FP + FN + TN == 1 for each label (normalized)."""
        y_true, y_pred = self._perfect_batch()
        result = compute_confusion_table(y_true, y_pred, ["A", "B"])
        for lbl in result.index:
            total = (
                result.loc[lbl, "TP"]
                + result.loc[lbl, "FP"]
                + result.loc[lbl, "FN"]
                + result.loc[lbl, "TN"]
            )
            assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _compute_misclassification_table
# ---------------------------------------------------------------------------


class TestComputeMisclassificationTable:
    """Tests for _compute_misclassification_table."""

    def test_output_shape(self):
        """Output has (n_labels+1) rows and (n_labels+2) columns (inc. fraction_time)."""
        n = 3
        m1 = np.eye(n, dtype=int)
        m2 = np.eye(n, dtype=int)
        result = _compute_misclassification_table(
            m1, m2, "true", "pred", ["A", "B", "C"]
        )
        assert result.shape == (
            n + 1,
            n + 2,
        )  # +1 NOLABEL row, +1 NOLABEL col, +1 fraction_time

    def test_diagonal_dominant_for_perfect_predictions(self):
        """Perfect label alignment → high diagonal values."""
        m = np.zeros((6, 2), dtype=int)
        m[:3, 0] = 1  # first 3 rows: label A
        m[3:, 1] = 1  # last 3 rows: label B
        result = _compute_misclassification_table(
            m, m.copy(), "true", "pred", ["A", "B"]
        )
        assert result.loc["true_A", "pred_A"] == pytest.approx(1.0)
        assert result.loc["true_B", "pred_B"] == pytest.approx(1.0)

    def test_column_and_index_names(self):
        """Columns and index follow <suffix>_<label> convention."""
        m = np.zeros((4, 2), dtype=int)
        result = _compute_misclassification_table(
            m, m.copy(), "true", "pred", ["X", "Y"]
        )
        assert "pred_X" in result.columns
        assert "true_X" in result.index


# ---------------------------------------------------------------------------
# compute_misclassification_tables
# ---------------------------------------------------------------------------


class TestComputeMisclassificationTables:
    """Tests for compute_misclassification_tables."""

    def test_returns_two_tables(self):
        """Returns dict with two DataFrames keyed by suffix combinations."""
        m1 = np.zeros((4, 2), dtype=int)
        m2 = np.zeros((4, 2), dtype=int)
        result = compute_misclassification_tables(m1, m2, "true", "pred", ["A", "B"])
        assert "true_pred" in result
        assert "pred_true" in result

    def test_both_values_are_dataframes(self):
        """Both returned values are DataFrames."""
        import pandas as pd

        m = np.zeros((4, 2), dtype=int)
        result = compute_misclassification_tables(
            m, m.copy(), "true", "pred", ["A", "B"]
        )
        for df in result.values():
            assert isinstance(df, pd.DataFrame)
