"""Tests for snippets module.

Tests for snippet table creation, filtering, and stats computation.

Created using: claude-sonnet-4-6 on 2026-03-31
"""

import json

import numpy as np
import pandas as pd
import pytest
import zarr

from orcai.snippets import (
    DATA_TYPES,
    _compute_snippet_stats,
    _filter_snippet_table,
    _make_snippet_table,
)


# ---------------------------------------------------------------------------
# _compute_snippet_stats
# ---------------------------------------------------------------------------


class TestComputeSnippetStats:
    """Tests for _compute_snippet_stats."""

    def test_columns_include_data_types_and_total(self, snippet_table_df, label_calls):
        """Output has train/val/test and total columns."""
        stats = _compute_snippet_stats(snippet_table_df, for_calls=label_calls)
        for col in DATA_TYPES + ["total"]:
            assert col in stats.columns

    def test_total_equals_sum_of_types(self, snippet_table_df, label_calls):
        """Total column equals sum of train + val + test."""
        stats = _compute_snippet_stats(snippet_table_df, for_calls=label_calls)
        computed_total = stats[["train", "val", "test"]].sum(axis=1)
        pd.testing.assert_series_equal(
            stats["total"], computed_total, check_names=False
        )

    def test_rows_are_call_names(self, snippet_table_df, label_calls):
        """One row per call in for_calls."""
        stats = _compute_snippet_stats(snippet_table_df, for_calls=label_calls)
        assert list(stats.index) == label_calls

    def test_equalizing_factor_columns_present(self, snippet_table_df, label_calls):
        """Equalizing factor columns (<type>_ef) are present."""
        stats = _compute_snippet_stats(snippet_table_df, for_calls=label_calls)
        for col in [f"{t}_ef" for t in DATA_TYPES]:
            assert col in stats.columns


# ---------------------------------------------------------------------------
# _filter_snippet_table
# ---------------------------------------------------------------------------


class TestFilterSnippetTable:
    """Tests for _filter_snippet_table."""

    def test_reduces_no_label_rows(self, snippet_table_df, orcai_parameter_snippets):
        """Fraction of no-label snippets decreases after filtering."""
        calls = orcai_parameter_snippets["calls"]
        n_no_label_before = (snippet_table_df[calls].sum(axis=1) <= 1e-7).sum()
        result = _filter_snippet_table(snippet_table_df, orcai_parameter_snippets)
        n_no_label_after = (result[calls].sum(axis=1) <= 1e-7).sum()
        assert n_no_label_after <= n_no_label_before

    def test_output_is_dataframe(self, snippet_table_df, orcai_parameter_snippets):
        """Returns a DataFrame."""
        result = _filter_snippet_table(snippet_table_df, orcai_parameter_snippets)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_preserved(self, snippet_table_df, orcai_parameter_snippets):
        """Output columns are a subset of input columns."""
        result = _filter_snippet_table(snippet_table_df, orcai_parameter_snippets)
        assert set(result.columns) == set(snippet_table_df.columns)

    def test_fraction_removal_zero_keeps_all(
        self, snippet_table_df, orcai_parameter_snippets
    ):
        """fraction_removal=0 keeps all no-label snippets."""
        params = {**orcai_parameter_snippets}
        params["snippets"] = {**params["snippets"], "fraction_removal": 0.0}
        result = _filter_snippet_table(snippet_table_df, params)
        assert len(result) == len(snippet_table_df)

    def test_index_reset(self, snippet_table_df, orcai_parameter_snippets):
        """Returned DataFrame has a contiguous integer index."""
        result = _filter_snippet_table(snippet_table_df, orcai_parameter_snippets)
        assert list(result.index) == list(range(len(result)))

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_deterministic_with_same_seed(
        self, snippet_table_df, orcai_parameter_snippets, seed
    ):
        """Same rng seed produces identical results."""
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        r1 = _filter_snippet_table(snippet_table_df, orcai_parameter_snippets, rng=rng1)
        r2 = _filter_snippet_table(snippet_table_df, orcai_parameter_snippets, rng=rng2)
        pd.testing.assert_frame_equal(
            r1.reset_index(drop=True), r2.reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# _make_snippet_table
# ---------------------------------------------------------------------------


def _build_recording_dir(
    base_path,
    label_names: list[str],
    n_time: int = 1000,
    recording_duration: float = 1000.0,
) -> None:
    """Create minimal directory structure for _make_snippet_table."""
    recording_dir = base_path
    spec_dir = recording_dir / "spectrogram"
    spec_dir.mkdir(parents=True)
    labels_dir = recording_dir / "labels"
    labels_dir.mkdir()

    # times.json
    times_data = {"min": 0.0, "max": recording_duration, "length": n_time}
    (spec_dir / "times.json").write_text(json.dumps(times_data))

    # label_list.json
    label_list = {lbl: "present" for lbl in label_names}
    (labels_dir / "label_list.json").write_text(json.dumps(label_list))

    # labels.zarr — random binary labels
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2, size=(n_time, len(label_names))).astype("float32")
    arr = zarr.open_array(
        labels_dir / "labels.zarr",
        mode="w",
        shape=(n_time, len(label_names)),
        chunks=(n_time, len(label_names)),
        dtype="float32",
    )
    arr[:] = data


class TestMakeSnippetTable:
    """Tests for _make_snippet_table."""

    def test_success_returns_dataframe(
        self, tmp_path, label_calls, orcai_parameter_snippets
    ):
        """Returns a DataFrame when directory structure is complete."""
        rec_dir = tmp_path / "test_rec"
        _build_recording_dir(rec_dir, label_calls)
        snippet_table, _, _, _, status = _make_snippet_table(
            rec_dir, orcai_parameter_snippets
        )
        assert status == "success"
        assert isinstance(snippet_table, pd.DataFrame)

    def test_output_columns(self, tmp_path, label_calls, orcai_parameter_snippets):
        """Snippet table contains required columns."""
        rec_dir = tmp_path / "test_rec"
        _build_recording_dir(rec_dir, label_calls)
        snippet_table, *_ = _make_snippet_table(rec_dir, orcai_parameter_snippets)
        for col in [
            "recording",
            "recording_data_dir",
            "data_type",
            "row_start",
            "row_stop",
        ]:
            assert col in snippet_table.columns

    def test_missing_spectrogram_raises(
        self, tmp_path, label_calls, orcai_parameter_snippets
    ):
        """FileNotFoundError raised when times.json is missing."""
        rec_dir = tmp_path / "no_spec"
        rec_dir.mkdir()
        (rec_dir / "labels").mkdir()
        (rec_dir / "labels" / "label_list.json").write_text(json.dumps({}))
        with pytest.raises(FileNotFoundError):
            _make_snippet_table(rec_dir, orcai_parameter_snippets)

    def test_missing_label_file_returns_none(
        self, tmp_path, label_calls, orcai_parameter_snippets
    ):
        """Returns None snippet table when labels.zarr is missing."""
        rec_dir = tmp_path / "no_labels"
        rec_dir.mkdir()
        spec_dir = rec_dir / "spectrogram"
        spec_dir.mkdir()
        (spec_dir / "times.json").write_text(
            json.dumps({"min": 0.0, "max": 500.0, "length": 500})
        )
        snippet_table, _, _, _, status = _make_snippet_table(
            rec_dir, orcai_parameter_snippets
        )
        assert snippet_table is None
        assert status == "missing label files"

    def test_recording_too_short_returns_none(
        self, tmp_path, label_calls, orcai_parameter_snippets
    ):
        """Returns None when recording is shorter than segment_duration."""
        rec_dir = tmp_path / "short_rec"
        # Recording of 5s, segment_duration=10 → n_segments=0
        _build_recording_dir(rec_dir, label_calls, n_time=50, recording_duration=5.0)
        snippet_table, _, _, _, status = _make_snippet_table(
            rec_dir, orcai_parameter_snippets
        )
        assert snippet_table is None
        assert status == "shorter than segment_duration"
