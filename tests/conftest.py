"""Pytest Configuration and Shared Fixtures

Shared pytest fixtures for OrcAI tests including sample inputs, labels, and model parameters.

Created using: claude-haiku-4.5 on 2026-03-31
Updated using: claude-sonnet-4-6 on 2026-03-31
"""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import zarr
from librosa import fft_frequencies

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


@pytest.fixture(scope="function")
def snippet_table_fixture():
    """Create a sample DataFrame with snippet table data."""
    # Create temporary zarr structure
    tmp_dir = tempfile.mkdtemp()
    base_path = Path(tmp_dir)

    # Create a test recording data directory with zarr files
    recording_dir = base_path / "recording_0"
    recording_dir.mkdir()

    spec_dir = recording_dir / "spectrogram"
    spec_dir.mkdir()
    labels_dir = recording_dir / "labels"
    labels_dir.mkdir()

    # Create zarr arrays
    spec_zarr = zarr.open_array(
        spec_dir / "spectrogram.zarr",
        mode="w",
        shape=(256, 64),
        chunks=(100, 64),
        dtype="float32",
    )
    spec_zarr[:] = np.random.randn(256, 64).astype("float32")

    labels_zarr = zarr.open_array(
        labels_dir / "labels.zarr",
        mode="w",
        shape=(256, 7),
        chunks=(100, 7),
        dtype="float32",
    )
    labels_zarr[:] = np.random.randint(0, 2, (256, 7)).astype("float32")

    # Create snippet table
    data = {
        "recording_data_dir": [str(recording_dir), str(recording_dir)],
        "row_start": [0, 128],
        "row_stop": [128, 256],
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def annotation_file_fixture(tmp_path):
    """Create a sample annotation file."""
    annotation_path = tmp_path / "test_annotation.txt"
    with open(annotation_path, "w") as f:
        f.write("0.5\t1.5\tBR\n")
        f.write("2.0\t3.0\tBUZZ\n")
        f.write("4.5\t5.5\tWHISTLE\n")
    return annotation_path



# ---------------------------------------------------------------------------
# Labels module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def label_calls():
    """Standard list of label calls for labels module tests."""
    return ["BR", "BUZZ", "WHISTLE"]


@pytest.fixture(scope="function")
def identity_eq(label_calls):
    """Identity call equivalences mapping each label name to itself.

    Required because _convert_annotation only creates the 'label' column
    when call_equivalences is provided.
    """
    return {lbl: lbl for lbl in label_calls}


@pytest.fixture(scope="function")
def labels_recording_fixture(tmp_path, label_calls):
    """Recording with spectrogram times.json and a standard annotation file.

    Creates:
    - tmp_path/<recording>/spectrogram/times.json  (0–10 s, 100 steps)
    - tmp_path/annotations/<recording>.txt          (label_calls[0] at 1–3 s,
                                                     label_calls[1] at 6–8 s)

    Returns a dict with keys: recording, recording_data_dir, annotation_file, ann_dir.
    """
    recording = "test_rec"
    spec_dir = tmp_path / recording / "spectrogram"
    spec_dir.mkdir(parents=True)
    (spec_dir / "times.json").write_text(
        json.dumps({"min": 0.0, "max": 10.0, "length": 100})
    )
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    ann_file = ann_dir / f"{recording}.txt"
    ann_file.write_text(f"1.0\t3.0\t{label_calls[0]}\n6.0\t8.0\t{label_calls[1]}")
    return {
        "recording": recording,
        "recording_data_dir": tmp_path,
        "annotation_file": ann_file,
        "ann_dir": ann_dir,
    }


@pytest.fixture(scope="function")
def recording_table_csv(tmp_path, label_calls, labels_recording_fixture):
    """Recording table CSV with a single recording, all labels set to True."""
    rec = labels_recording_fixture["recording"]
    ann_dir = labels_recording_fixture["ann_dir"]
    row: dict = {
        "recording": rec,
        "base_dir_annotation": str(ann_dir),
        "rel_annotation_path": f"{rec}.txt",
    }
    for lbl in label_calls:
        row[lbl] = True
    csv_path = tmp_path / "recording_table.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Spectrogram module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def spectrogram_parameter():
    """Minimal spectrogram parameters for fast unit tests (low sample rate)."""
    return {
        "sampling_rate": 8000,
        "nfft": 512,
        "n_overlap": 256,
        "freq_range": [200, 3000],
        "quantiles": [0.01, 0.99],
    }


@pytest.fixture(scope="function")
def spectrogram_frequencies(spectrogram_parameter):
    """Frequency axis matching spectrogram_parameter (via librosa)."""
    return fft_frequencies(
        sr=spectrogram_parameter["sampling_rate"],
        n_fft=spectrogram_parameter["nfft"],
    )


@pytest.fixture(scope="function")
def synthetic_spectrogram_raw(spectrogram_parameter):
    """Synthetic raw dB spectrogram (freq x time) matching spectrogram_parameter."""
    n_freq = spectrogram_parameter["nfft"] // 2 + 1
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_freq, 50)).astype(np.float32)


# ---------------------------------------------------------------------------
# Predict module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def predicted_labels_df():
    """Sample DataFrame of predicted labels with '*' suffix."""
    return pd.DataFrame(
        {
            "start": [0, 10, 20, 50],
            "stop": [5, 15, 25, 55],
            "label": ["BR*", "BUZZ*", "BR*", "WHISTLE*"],
            "mean_p": [0.8, 0.9, 0.7, 0.6],
            "label_source": ["test-model"] * 4,
        }
    )


@pytest.fixture(scope="function")
def call_duration_limits_dict():
    """Call duration limits dict (seconds) keyed by label name (no suffix)."""
    return {
        "BR": [2.0, 8.0],
        "BUZZ": [3.0, 20.0],
        "WHISTLE": [1.0, 10.0],
    }


# ---------------------------------------------------------------------------
# Snippets module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def snippet_table_df(label_calls):
    """Minimal snippet table for _compute_snippet_stats / _filter_snippet_table tests."""
    rows = []
    for i, data_type in enumerate(["train", "val", "test"] * 4):
        row: dict = {
            "recording": f"rec{i}",
            "recording_data_dir": f"/data/rec{i}",
            "data_type": data_type,
            "row_start": i * 100,
            "row_stop": i * 100 + 100,
        }
        for j, lbl in enumerate(label_calls):
            row[lbl] = float((i + j) % 3)  # mix of 0.0, 1.0, 2.0 values
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="function")
def orcai_parameter_snippets(label_calls):
    """orcai_parameter with model and snippets sections for snippet module tests."""
    return {
        "name": "test",
        "seed": 42,
        "calls": label_calls,
        "model": {
            "filters": [32, 64],
            "n_batch_train": 2,
            "n_batch_val": 1,
            "n_batch_test": 1,
            "batch_size": 4,
        },
        "snippets": {
            "segment_duration": 100,  # large enough so val/test slices fit snippet_duration
            "snippets_per_sec": 0.2,
            "snippet_duration": 4,
            "fraction_removal": 0.5,
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
    }
