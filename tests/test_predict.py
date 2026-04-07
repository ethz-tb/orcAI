"""Tests for predict module.

Tests for prediction helper functions: duration checking, binary prediction
computation, label DataFrame generation, and prediction filtering.

Created using: claude-sonnet-4-6 on 2026-03-31
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orcai.predict import (
    _calulate_mean_probabilities,
    _check_duration,
    _generate_label_dataframe,
    _get_output_path,
    compute_binary_predictions,
    filter_predictions,
    filter_predictions_file,
)


# ---------------------------------------------------------------------------
# _check_duration
# ---------------------------------------------------------------------------


class TestCheckDuration:
    """Tests for _check_duration."""

    def _row(self, label: str, duration: float, suffix: str = "*") -> pd.Series:
        """Build a minimal row Series for _check_duration."""
        return pd.Series({"label": label + suffix, "duration": duration})

    def test_keep_within_limits(self, call_duration_limits_dict):
        """Returns 'keep' when duration is within limits."""
        row = self._row("BR", 5.0)  # duration 5s, delta_t=1 → 5s, limits [2, 8]
        result = _check_duration(row, call_duration_limits_dict, delta_t=1.0)
        assert result == "keep"

    def test_too_short(self, call_duration_limits_dict):
        """Returns 'too short' when duration * delta_t < min_duration."""
        row = self._row("BR", 1.0)  # 1s < min 2s
        result = _check_duration(row, call_duration_limits_dict, delta_t=1.0)
        assert result == "too short"

    def test_too_long(self, call_duration_limits_dict):
        """Returns 'too long' when duration * delta_t > max_duration."""
        row = self._row("BR", 10.0)  # 10s > max 8s
        result = _check_duration(row, call_duration_limits_dict, delta_t=1.0)
        assert result == "too long"

    def test_delta_t_scaling(self, call_duration_limits_dict):
        """delta_t scales duration correctly (duration_steps * delta_t vs limits)."""
        # BR limits [2, 8]. duration=3 steps * delta_t=0.5 = 1.5s → too short
        row = self._row("BR", 3.0)
        result = _check_duration(row, call_duration_limits_dict, delta_t=0.5)
        assert result == "too short"

    def test_default_limits_used_for_unknown_label(self):
        """Falls back to 'default' entry when label not in call_duration_limits."""
        limits = {"default": [1.0, 5.0]}
        row = self._row("UNKNOWN", 3.0)
        result = _check_duration(row, limits, delta_t=1.0)
        assert result == "keep"

    def test_no_limits_always_keep(self):
        """Returns 'keep' when neither label nor 'default' is in limits dict."""
        row = self._row("BR", 100.0)
        result = _check_duration(row, {}, delta_t=1.0)
        assert result == "keep"

    def test_none_max_treated_as_infinity(self):
        """None max_duration is treated as inf."""
        limits = {"BR": [0.0, None]}
        row = self._row("BR", 9999.0)
        result = _check_duration(row, limits, delta_t=1.0)
        assert result == "keep"

    def test_none_min_treated_as_zero(self):
        """None min_duration is treated as 0."""
        limits = {"BR": [None, 100.0]}
        row = self._row("BR", 0.001)
        result = _check_duration(row, limits, delta_t=1.0)
        assert result == "keep"


# ---------------------------------------------------------------------------
# _calulate_mean_probabilities
# ---------------------------------------------------------------------------


class TestCalculateMeanProbabilities:
    """Tests for _calulate_mean_probabilities."""

    def test_single_point_call(self):
        """When start == stop, returns the single probability value."""
        probs = np.array([0.1, 0.9, 0.5])
        result = _calulate_mean_probabilities(probs, [1], [1])
        assert len(result) == 1
        assert result[0] == pytest.approx(0.9)

    def test_interval_mean(self):
        """Returns mean of slice [start:stop] for an interval call."""
        probs = np.array([0.2, 0.4, 0.6, 0.8])
        result = _calulate_mean_probabilities(probs, [1], [3])
        assert result[0] == pytest.approx(0.5)

    def test_multiple_calls(self):
        """Handles multiple calls with different start/stop indices."""
        probs = np.array([0.1, 0.5, 0.9, 0.3])
        result = _calulate_mean_probabilities(probs, [0, 2], [2, 4])
        assert len(result) == 2
        assert result[0] == pytest.approx(0.3)  # mean(probs[0:2]) = mean(0.1, 0.5)
        assert result[1] == pytest.approx(0.6)  # mean(0.9, 0.3)


# ---------------------------------------------------------------------------
# compute_binary_predictions
# ---------------------------------------------------------------------------


class TestComputeBinaryPredictions:
    """Tests for compute_binary_predictions."""

    def test_above_threshold_detected(self):
        """Consecutive above-threshold steps are returned as a call."""
        preds = np.zeros((10, 2))
        preds[2:5, 0] = 0.8  # BR active at steps 2-4
        starts, stops, labels, probs = compute_binary_predictions(
            preds, ["BR", "BUZZ"], threshold=0.5
        )
        assert "BR" in labels
        assert "BUZZ" not in labels

    def test_below_threshold_ignored(self):
        """Steps below threshold produce no detections."""
        preds = np.full((10, 2), 0.3)
        starts, stops, labels, probs = compute_binary_predictions(
            preds, ["BR", "BUZZ"], threshold=0.5
        )
        assert len(labels) == 0

    def test_multiple_labels_detected(self):
        """Multiple active labels are each returned."""
        preds = np.zeros((10, 2))
        preds[1:3, 0] = 0.9  # BR
        preds[5:8, 1] = 0.9  # BUZZ
        starts, stops, labels, probs = compute_binary_predictions(
            preds, ["BR", "BUZZ"], threshold=0.5
        )
        assert "BR" in labels
        assert "BUZZ" in labels

    def test_mean_probability_in_range(self):
        """Mean probabilities are within [0, 1]."""
        preds = np.zeros((10, 1))
        preds[3:7, 0] = 0.7
        _, _, _, probs = compute_binary_predictions(preds, ["BR"], threshold=0.5)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_output_lengths_consistent(self):
        """All output lists have the same length."""
        preds = np.zeros((20, 3))
        preds[2:5, 0] = 0.8
        preds[10:15, 2] = 0.9
        starts, stops, labels, probs = compute_binary_predictions(
            preds, ["BR", "BUZZ", "WHISTLE"], threshold=0.5
        )
        assert len(starts) == len(stops) == len(labels) == len(probs)

    def test_custom_threshold(self):
        """Custom threshold correctly gates detection."""
        preds = np.zeros((10, 1))
        preds[3:6, 0] = 0.6
        # threshold=0.7 → nothing detected
        _, _, labels_high, _ = compute_binary_predictions(preds, ["BR"], threshold=0.7)
        assert len(labels_high) == 0
        # threshold=0.5 → detected
        _, _, labels_low, _ = compute_binary_predictions(preds, ["BR"], threshold=0.5)
        assert "BR" in labels_low


# ---------------------------------------------------------------------------
# _generate_label_dataframe
# ---------------------------------------------------------------------------


class TestGenerateLabelDataframe:
    """Tests for _generate_label_dataframe."""

    def test_basic_output_columns(self):
        """Output DataFrame has start, stop, label, mean_p columns."""
        df = _generate_label_dataframe(
            row_starts=[0, 10],
            row_stops=[5, 15],
            label_names=["BR", "BUZZ"],
            mean_probabilities=[0.8, 0.9],
            time_steps_per_output_step=4,
            label_suffix="*",
        )
        assert set(df.columns) >= {"start", "stop", "label", "mean_p"}

    def test_time_step_scaling(self):
        """start/stop are scaled by time_steps_per_output_step."""
        df = _generate_label_dataframe(
            row_starts=[1],
            row_stops=[3],
            label_names=["BR"],
            mean_probabilities=[0.8],
            time_steps_per_output_step=8,
            label_suffix=None,
        )
        assert df.iloc[0]["start"] == 8
        assert df.iloc[0]["stop"] == 24

    def test_suffix_appended_to_label(self):
        """label_suffix is appended to each label name."""
        df = _generate_label_dataframe(
            row_starts=[0],
            row_stops=[5],
            label_names=["BR"],
            mean_probabilities=[0.8],
            time_steps_per_output_step=1,
            label_suffix="*",
        )
        assert df.iloc[0]["label"] == "BR*"

    def test_no_suffix_when_none(self):
        """label_suffix=None leaves label names unchanged."""
        df = _generate_label_dataframe(
            row_starts=[0],
            row_stops=[5],
            label_names=["BR"],
            mean_probabilities=[0.8],
            time_steps_per_output_step=1,
            label_suffix=None,
        )
        assert df.iloc[0]["label"] == "BR"

    def test_sorted_by_start(self):
        """Output is sorted by start position."""
        df = _generate_label_dataframe(
            row_starts=[10, 2],
            row_stops=[15, 6],
            label_names=["BUZZ", "BR"],
            mean_probabilities=[0.9, 0.7],
            time_steps_per_output_step=1,
            label_suffix=None,
        )
        assert list(df["start"]) == sorted(df["start"])


# ---------------------------------------------------------------------------
# _get_output_path
# ---------------------------------------------------------------------------


class TestGetOutputPath:
    """Tests for _get_output_path."""

    def test_output_path_format(self, tmp_path):
        """Output path follows <stem>_c<channel>_<model_name>_predicted.txt."""
        rec = tmp_path / "recording.wav"
        result = _get_output_path(rec, channel=1, model_name="mymodel")
        assert result.name == "recording_c1_mymodel_predicted.txt"
        assert result.parent == tmp_path

    def test_different_channel(self, tmp_path):
        """Channel number is embedded in the filename."""
        rec = tmp_path / "rec.wav"
        result = _get_output_path(rec, channel=2, model_name="m")
        assert "_c2_" in result.name


# ---------------------------------------------------------------------------
# filter_predictions
# ---------------------------------------------------------------------------


class TestFilterPredictions:
    """Tests for filter_predictions."""

    def test_keeps_all_within_limits(
        self, predicted_labels_df, call_duration_limits_dict
    ):
        """All calls within limits are kept."""
        # predicted_labels_df has durations 5, 5, 5, 5 (stop-start), delta_t=1
        # BR limits [2,8], BUZZ limits [3,20], WHISTLE limits [1,10] → all kept
        result = filter_predictions(
            predicted_labels_df,
            delta_t=1.0,
            call_duration_limits=call_duration_limits_dict,
        )
        assert len(result) == len(predicted_labels_df)

    def test_removes_too_short(self, call_duration_limits_dict):
        """Calls shorter than min_duration are removed."""
        df = pd.DataFrame(
            {
                "start": [0],
                "stop": [1],  # duration 1s, BR min is 2s → too short
                "label": ["BR*"],
                "mean_p": [0.9],
            }
        )
        result = filter_predictions(
            df, delta_t=1.0, call_duration_limits=call_duration_limits_dict
        )
        assert len(result) == 0

    def test_removes_too_long(self, call_duration_limits_dict):
        """Calls longer than max_duration are removed."""
        df = pd.DataFrame(
            {
                "start": [0],
                "stop": [20],  # duration 20s, BR max is 8s → too long
                "label": ["BR*"],
                "mean_p": [0.9],
            }
        )
        result = filter_predictions(
            df, delta_t=1.0, call_duration_limits=call_duration_limits_dict
        )
        assert len(result) == 0

    def test_empty_input_returns_empty(self, call_duration_limits_dict):
        """Empty input DataFrame is returned unchanged."""
        df = pd.DataFrame(columns=["start", "stop", "label", "mean_p"])
        result = filter_predictions(
            df, delta_t=1.0, call_duration_limits=call_duration_limits_dict
        )
        assert result.empty

    def test_output_columns_preserved(
        self, predicted_labels_df, call_duration_limits_dict
    ):
        """Output has the same columns as input (filter_predictions modifies df in-place)."""
        original_cols = list(predicted_labels_df.columns)
        result = filter_predictions(
            predicted_labels_df,
            delta_t=1.0,
            call_duration_limits=call_duration_limits_dict,
        )
        assert list(result.columns) == original_cols

    def test_limits_from_file(self, predicted_labels_df, tmp_path):
        """call_duration_limits can be given as a path to a JSON file."""
        limits = {"BR": [2.0, 8.0], "BUZZ": [3.0, 20.0], "WHISTLE": [1.0, 10.0]}
        limits_path = tmp_path / "limits.json"
        limits_path.write_text(json.dumps(limits))
        result = filter_predictions(
            predicted_labels_df, delta_t=1.0, call_duration_limits=limits_path
        )
        assert len(result) == len(predicted_labels_df)


# ---------------------------------------------------------------------------
# filter_predictions_file
# ---------------------------------------------------------------------------


class TestFilterPredictionsFile:
    """Tests for filter_predictions_file."""

    def _write_predictions_file(self, path: Path, rows: list[tuple]) -> None:
        """Write a tab-separated predictions file."""
        lines = "\n".join(
            f"{start}\t{stop}\t{label}\t{p}\tsource" for start, stop, label, p in rows
        )
        path.write_text(lines)

    def test_creates_filtered_file(self, tmp_path):
        """Filtered output file is created next to the input by default."""
        pred_file = tmp_path / "recording_predicted.txt"
        self._write_predictions_file(pred_file, [(0, 5, "BR*", 0.8)])
        limits = {"BR": [2.0, 8.0]}

        filter_predictions_file(pred_file, call_duration_limits=limits)

        out = tmp_path / "recording_predicted_filtered.txt"
        assert out.exists()

    def test_raises_if_output_exists_no_overwrite(self, tmp_path):
        """FileExistsError raised if output file already exists and overwrite=False."""
        pred_file = tmp_path / "r_predicted.txt"
        self._write_predictions_file(pred_file, [(0, 5, "BR*", 0.8)])
        out_file = tmp_path / "r_predicted_filtered.txt"
        out_file.write_text("existing")

        with pytest.raises(FileExistsError):
            filter_predictions_file(pred_file, call_duration_limits={}, overwrite=False)

    def test_overwrites_when_flag_set(self, tmp_path):
        """Existing output file is overwritten when overwrite=True."""
        pred_file = tmp_path / "r_predicted.txt"
        self._write_predictions_file(pred_file, [(0, 5, "BR*", 0.8)])
        out_file = tmp_path / "r_predicted_filtered.txt"
        out_file.write_text("old content")

        filter_predictions_file(
            pred_file, call_duration_limits={"BR": [2.0, 8.0]}, overwrite=True
        )
        assert out_file.read_text() != "old content"
