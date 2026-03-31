"""Tests for labels module.

Tests for _convert_annotation and create_label_arrays functions.

Created using: claude-sonnet-4-6 on 2026-03-31
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from orcai.auxiliary import MASK_VALUE
from orcai.labels import _convert_annotation, create_label_arrays


# ---------------------------------------------------------------------------
# Local helpers (not fixtures — used to create ad-hoc structures in tests
# that need non-standard setups not covered by conftest fixtures)
# ---------------------------------------------------------------------------


def _write_times_json(path: Path, t_min: float, t_max: float, length: int) -> None:
    """Write a times.json file; parent dirs created if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"min": t_min, "max": t_max, "length": length}))


def _write_annotation_file(path: Path, rows: list[tuple]) -> None:
    """Write a tab-separated annotation file with (start, stop, label) rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{s}\t{e}\t{lbl}" for s, e, lbl in rows))


# ---------------------------------------------------------------------------
# _convert_annotation
# ---------------------------------------------------------------------------


class TestConvertAnnotation:
    """Tests for _convert_annotation."""

    def test_present_labels_binary(
        self, labels_recording_fixture, label_calls, identity_eq
    ):
        """Labels in labels_present get 0/1 values from annotation intervals."""
        ann = labels_recording_fixture["annotation_file"]
        rec_data_dir = labels_recording_fixture["recording_data_dir"]

        df, _ = _convert_annotation(
            annotation_file_path=ann,
            recording_data_dir=rec_data_dir,
            label_calls=label_calls,
            labels_present=[label_calls[0], label_calls[1]],
            labels_masked=[label_calls[2]],
            call_equivalences=identity_eq,
        )

        t_vec = np.linspace(0.0, 10.0, 100)
        np.testing.assert_array_equal(
            df[label_calls[0]].values, ((t_vec >= 1.0) & (t_vec <= 3.0)).astype(int)
        )
        np.testing.assert_array_equal(
            df[label_calls[1]].values, ((t_vec >= 6.0) & (t_vec <= 8.0)).astype(int)
        )

    def test_masked_labels_have_mask_value(
        self, labels_recording_fixture, label_calls, identity_eq
    ):
        """Labels in labels_masked are filled with MASK_VALUE."""
        ann = labels_recording_fixture["annotation_file"]
        rec_data_dir = labels_recording_fixture["recording_data_dir"]

        df, _ = _convert_annotation(
            annotation_file_path=ann,
            recording_data_dir=rec_data_dir,
            label_calls=label_calls,
            labels_present=[label_calls[0]],
            labels_masked=label_calls[1:],
            call_equivalences=identity_eq,
        )

        for lbl in label_calls[1:]:
            assert (df[lbl] == MASK_VALUE).all()

    def test_column_order_matches_label_calls(
        self, labels_recording_fixture, label_calls, identity_eq
    ):
        """Output DataFrame columns are in label_calls order."""
        ann = labels_recording_fixture["annotation_file"]
        rec_data_dir = labels_recording_fixture["recording_data_dir"]

        df, _ = _convert_annotation(
            annotation_file_path=ann,
            recording_data_dir=rec_data_dir,
            label_calls=label_calls,
            labels_present=[label_calls[0]],
            labels_masked=label_calls[1:],
            call_equivalences=identity_eq,
        )

        assert list(df.columns) == label_calls

    def test_label_dict_present_and_masked(
        self, labels_recording_fixture, label_calls, identity_eq
    ):
        """Returned label_dict reflects present/masked status correctly."""
        ann = labels_recording_fixture["annotation_file"]
        rec_data_dir = labels_recording_fixture["recording_data_dir"]

        _, label_dict = _convert_annotation(
            annotation_file_path=ann,
            recording_data_dir=rec_data_dir,
            label_calls=label_calls,
            labels_present=[label_calls[0]],
            labels_masked=label_calls[1:],
            call_equivalences=identity_eq,
        )

        assert label_dict[label_calls[0]] == "present"
        for lbl in label_calls[1:]:
            assert label_dict[lbl] == "masked"

    def test_label_dict_order_matches_label_calls(
        self, labels_recording_fixture, label_calls, identity_eq
    ):
        """label_dict keys are in label_calls order."""
        ann = labels_recording_fixture["annotation_file"]
        rec_data_dir = labels_recording_fixture["recording_data_dir"]

        _, label_dict = _convert_annotation(
            annotation_file_path=ann,
            recording_data_dir=rec_data_dir,
            label_calls=label_calls,
            labels_present=[label_calls[0]],
            labels_masked=label_calls[1:],
            call_equivalences=identity_eq,
        )

        assert list(label_dict.keys()) == label_calls

    def test_missing_spectrogram_raises(self, tmp_path, label_calls, identity_eq):
        """FileNotFoundError raised when times.json does not exist."""
        ann = tmp_path / "ann" / "missing_rec.txt"
        _write_annotation_file(ann, [(1.0, 2.0, label_calls[0])])

        with pytest.raises(FileNotFoundError):
            _convert_annotation(
                annotation_file_path=ann,
                recording_data_dir=tmp_path,
                label_calls=label_calls,
                labels_present=[label_calls[0]],
                labels_masked=label_calls[1:],
                call_equivalences=identity_eq,
            )

    def test_call_equivalences_dict(self, tmp_path, label_calls):
        """Call equivalences dict remaps annotation labels before processing."""
        recording = "rec_eq_dict"
        _write_times_json(
            tmp_path / recording / "spectrogram" / "times.json", 0.0, 10.0, 100
        )
        ann = tmp_path / "ann" / f"{recording}.txt"
        _write_annotation_file(ann, [(1.0, 3.0, "CALL_A")])

        df, _ = _convert_annotation(
            annotation_file_path=ann,
            recording_data_dir=tmp_path,
            label_calls=label_calls,
            labels_present=[label_calls[0]],
            labels_masked=label_calls[1:],
            call_equivalences={"CALL_A": label_calls[0]},
        )

        t_vec = np.linspace(0.0, 10.0, 100)
        np.testing.assert_array_equal(
            df[label_calls[0]].values,
            ((t_vec >= 1.0) & (t_vec <= 3.0)).astype(int),
        )

    def test_call_equivalences_file(self, tmp_path, label_calls):
        """Call equivalences are loaded from a JSON file when a path is given."""
        recording = "rec_eq_file"
        _write_times_json(
            tmp_path / recording / "spectrogram" / "times.json", 0.0, 10.0, 100
        )
        ann = tmp_path / "ann" / f"{recording}.txt"
        _write_annotation_file(ann, [(2.0, 4.0, "ORIG")])

        eq_file = tmp_path / "equivalences.json"
        eq_file.write_text(json.dumps({"ORIG": label_calls[1]}))

        df, _ = _convert_annotation(
            annotation_file_path=ann,
            recording_data_dir=tmp_path,
            label_calls=label_calls,
            labels_present=[label_calls[1]],
            labels_masked=[label_calls[0], label_calls[2]],
            call_equivalences=eq_file,
        )

        t_vec = np.linspace(0.0, 10.0, 100)
        np.testing.assert_array_equal(
            df[label_calls[1]].values,
            ((t_vec >= 2.0) & (t_vec <= 4.0)).astype(int),
        )

    def test_output_length_matches_time_vector(
        self, tmp_path, label_calls, identity_eq
    ):
        """DataFrame row count equals the length specified in times.json."""
        recording = "rec_length"
        _write_times_json(
            tmp_path / recording / "spectrogram" / "times.json", 0.0, 10.0, 50
        )
        ann = tmp_path / "ann" / f"{recording}.txt"
        _write_annotation_file(ann, [(1.0, 2.0, label_calls[0])])

        df, _ = _convert_annotation(
            annotation_file_path=ann,
            recording_data_dir=tmp_path,
            label_calls=label_calls,
            labels_present=[label_calls[0]],
            labels_masked=label_calls[1:],
            call_equivalences=identity_eq,
        )

        assert len(df) == 50


# ---------------------------------------------------------------------------
# create_label_arrays
# ---------------------------------------------------------------------------


class TestCreateLabelArrays:
    """Tests for create_label_arrays."""

    def test_creates_zarr_and_label_list(
        self, recording_table_csv, labels_recording_fixture, label_calls, identity_eq
    ):
        """zarr array and label_list.json are written for each recording."""
        output_dir = labels_recording_fixture["recording_data_dir"]
        recording = labels_recording_fixture["recording"]

        create_label_arrays(
            recording_table_csv,
            output_dir,
            orcai_parameter={"calls": label_calls},
            call_equivalences=identity_eq,
        )

        labels_dir = output_dir / recording / "labels"
        assert (labels_dir / "labels.zarr").exists()
        assert (labels_dir / "label_list.json").exists()

    def test_zarr_shape(self, tmp_path, label_calls, identity_eq):
        """Saved zarr has shape (time_length, n_labels)."""
        recording = "rec_shape"
        _write_times_json(
            tmp_path / recording / "spectrogram" / "times.json", 0.0, 10.0, 80
        )
        ann_dir = tmp_path / "annotations"
        _write_annotation_file(
            ann_dir / f"{recording}.txt", [(1.0, 2.0, label_calls[0])]
        )

        row: dict = {
            "recording": recording,
            "base_dir_annotation": str(ann_dir),
            "rel_annotation_path": f"{recording}.txt",
        }
        for lbl in label_calls:
            row[lbl] = True
        csv = tmp_path / "table.csv"
        pd.DataFrame([row]).to_csv(csv, index=False)

        create_label_arrays(
            csv,
            tmp_path,
            orcai_parameter={"calls": label_calls},
            call_equivalences=identity_eq,
        )

        arr = zarr.open_array(tmp_path / recording / "labels" / "labels.zarr", mode="r")
        assert arr.shape == (80, len(label_calls))

    def test_skip_existing_labels_no_overwrite(
        self, recording_table_csv, labels_recording_fixture, label_calls, identity_eq
    ):
        """Existing labels directory is not overwritten when overwrite=False."""
        output_dir = labels_recording_fixture["recording_data_dir"]
        recording = labels_recording_fixture["recording"]

        labels_dir = output_dir / recording / "labels"
        labels_dir.mkdir(parents=True)
        sentinel = labels_dir / "sentinel.txt"
        sentinel.write_text("untouched")

        create_label_arrays(
            recording_table_csv,
            output_dir,
            orcai_parameter={"calls": label_calls},
            call_equivalences=identity_eq,
            overwrite=False,
        )

        assert sentinel.exists(), "sentinel deleted — existing labels were overwritten"

    def test_overwrite_replaces_labels(
        self, recording_table_csv, labels_recording_fixture, label_calls, identity_eq
    ):
        """Existing labels are replaced when overwrite=True."""
        output_dir = labels_recording_fixture["recording_data_dir"]
        recording = labels_recording_fixture["recording"]
        labels_dir = output_dir / recording / "labels"

        create_label_arrays(
            recording_table_csv,
            output_dir,
            orcai_parameter={"calls": label_calls},
            call_equivalences=identity_eq,
            overwrite=True,
        )
        assert (labels_dir / "labels.zarr").exists()

        create_label_arrays(
            recording_table_csv,
            output_dir,
            orcai_parameter={"calls": label_calls},
            call_equivalences=identity_eq,
            overwrite=True,
        )
        arr = zarr.open_array(labels_dir / "labels.zarr", mode="r")
        assert arr.shape[1] == len(label_calls)

    def test_skip_recording_with_no_labels_present(
        self, labels_recording_fixture, label_calls, identity_eq, tmp_path
    ):
        """Recordings where all label columns are False produce no output."""
        output_dir = labels_recording_fixture["recording_data_dir"]
        recording = labels_recording_fixture["recording"]
        ann_dir = labels_recording_fixture["ann_dir"]

        row: dict = {
            "recording": recording,
            "base_dir_annotation": str(ann_dir),
            "rel_annotation_path": f"{recording}.txt",
        }
        for lbl in label_calls:
            row[lbl] = False
        csv = tmp_path / "table_no_labels.csv"
        pd.DataFrame([row]).to_csv(csv, index=False)

        create_label_arrays(
            csv,
            output_dir,
            orcai_parameter={"calls": label_calls},
            call_equivalences=identity_eq,
        )

        assert not (output_dir / recording / "labels").exists()

    def test_skip_recording_missing_annotation(
        self, tmp_path, label_calls, identity_eq
    ):
        """Rows with NaN base_dir_annotation are skipped without error."""
        recording = "rec_no_ann"
        _write_times_json(
            tmp_path / recording / "spectrogram" / "times.json", 0.0, 10.0, 100
        )
        row: dict = {
            "recording": recording,
            "base_dir_annotation": float("nan"),
            "rel_annotation_path": f"{recording}.txt",
        }
        for lbl in label_calls:
            row[lbl] = True
        csv = tmp_path / "table_no_ann.csv"
        pd.DataFrame([row]).to_csv(csv, index=False)

        create_label_arrays(
            csv,
            tmp_path,
            orcai_parameter={"calls": label_calls},
            call_equivalences=identity_eq,
        )

        assert not (tmp_path / recording / "labels").exists()

    def test_base_dir_annotation_override(
        self,
        recording_table_csv,
        labels_recording_fixture,
        label_calls,
        identity_eq,
        tmp_path,
    ):
        """base_dir_annotation parameter overrides the table column."""
        output_dir = labels_recording_fixture["recording_data_dir"]
        recording = labels_recording_fixture["recording"]
        correct_ann_dir = labels_recording_fixture["ann_dir"]

        # Overwrite the table to point to a wrong (non-existent) dir
        wrong_dir = tmp_path / "wrong_ann"
        row: dict = {
            "recording": recording,
            "base_dir_annotation": str(wrong_dir),
            "rel_annotation_path": f"{recording}.txt",
        }
        for lbl in label_calls:
            row[lbl] = True
        csv = tmp_path / "table_wrong_dir.csv"
        pd.DataFrame([row]).to_csv(csv, index=False)

        create_label_arrays(
            csv,
            output_dir,
            base_dir_annotation=correct_ann_dir,
            orcai_parameter={"calls": label_calls},
            call_equivalences=identity_eq,
        )

        assert (output_dir / recording / "labels" / "labels.zarr").exists()
