"""Tests for io module.

Comprehensive pytest tests for data loading, saving, and model I/O functions.

Created using: claude-haiku-4.5 on 2026-03-31
"""

import json

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import zarr

from orcai.io import (
    DataLoader,
    _convert_steps_to_seconds,
    _parse_example,
    _serialize_example,
    generate_times_from_spectrogram,
    load_dataset,
    read_annotation_file,
    read_json,
    save_as_zarr,
    save_dataset,
    save_predictions,
    save_prediction_probabilities,
    write_json,
    write_vector_to_json,
)


class TestJsonIO:
    """Tests for JSON read/write functions."""

    def test_write_and_read_json(self, tmp_path):
        """Test writing and reading JSON data."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        json_path = tmp_path / "test.json"

        write_json(data, json_path)
        loaded = read_json(json_path)

        assert loaded == data

    def test_write_and_read_nested_json(self, tmp_path):
        """Test writing and reading nested JSON data."""
        data = {
            "nested": {"level1": {"level2": [1, 2, 3]}},
            "array": [{"a": 1}, {"b": 2}],
        }
        json_path = tmp_path / "nested.json"

        write_json(data, json_path)
        loaded = read_json(json_path)

        assert loaded == data

    def test_write_json_creates_file(self, tmp_path):
        """Test that write_json creates the file."""
        data = {"test": "data"}
        json_path = tmp_path / "new_file.json"

        assert not json_path.exists()
        write_json(data, json_path)
        assert json_path.exists()

    def test_read_json_file_not_found(self, tmp_path):
        """Test read_json raises error for non-existent file."""
        json_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            read_json(json_path)

    def test_write_and_read_empty_dict(self, tmp_path):
        """Test writing and reading empty dictionary."""
        data = {}
        json_path = tmp_path / "empty.json"

        write_json(data, json_path)
        loaded = read_json(json_path)

        assert loaded == {}


class TestVectorIO:
    """Tests for vector I/O functions."""

    def test_write_vector_to_json(self, tmp_path):
        """Test writing vector to JSON."""
        vector = np.linspace(0, 10, 100)
        json_path = tmp_path / "vector.json"

        write_vector_to_json(vector, json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert data["min"] == pytest.approx(0.0)
        assert data["max"] == pytest.approx(10.0)
        assert data["length"] == 100

    def test_write_vector_list_to_json(self, tmp_path):
        """Test writing list vector to JSON."""
        vector = [0, 5, 10]
        json_path = tmp_path / "vector.json"

        write_vector_to_json(vector, json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert data["min"] == 0
        assert data["max"] == 10
        assert data["length"] == 3

    def test_generate_times_from_spectrogram(self, tmp_path):
        """Test generating times from spectrogram JSON."""
        vector = np.linspace(0, 10, 100)
        json_path = tmp_path / "vector.json"

        write_vector_to_json(vector, json_path)
        generated = generate_times_from_spectrogram(json_path)

        assert len(generated) == 100
        assert generated[0] == pytest.approx(0.0)
        assert generated[-1] == pytest.approx(10.0)
        np.testing.assert_array_almost_equal(generated, vector)


class TestZarrIO:
    """Tests for Zarr I/O functions."""

    def test_save_as_zarr(self, tmp_path):
        """Test saving array to Zarr."""
        data = np.random.randn(100, 50).astype("float32")
        zarr_path = tmp_path / "test.zarr"

        save_as_zarr(data, zarr_path)

        assert zarr_path.exists()
        loaded = np.array(zarr.open_array(zarr_path, mode="r")).astype("float32")
        np.testing.assert_array_almost_equal(loaded, data)

    def test_save_as_zarr_shape(self, tmp_path):
        """Test that saved Zarr has correct shape."""
        data = np.random.randn(200, 100).astype("float32")
        zarr_path = tmp_path / "test.zarr"

        save_as_zarr(data, zarr_path)

        zarr_array = zarr.open_array(zarr_path, mode="r")
        assert zarr_array.shape == (200, 100)


class TestSerializationFunctions:
    """Tests for TFRecord serialization functions."""

    def test_serialize_example(self):
        """Test serializing example to TFRecord format."""
        spectrogram = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        labels = tf.constant([[0.0, 1.0]], dtype=tf.float32)

        serialized = _serialize_example(spectrogram, labels)

        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_serialize_example_different_shapes(self):
        """Test serializing examples with different shapes."""
        spec1 = tf.random.normal((100, 64))
        labels1 = tf.random.uniform((50, 7))

        spec2 = tf.random.normal((200, 64))
        labels2 = tf.random.uniform((100, 7))

        ser1 = _serialize_example(spec1, labels1)
        ser2 = _serialize_example(spec2, labels2)

        assert isinstance(ser1, bytes)
        assert isinstance(ser2, bytes)
        # Different data should produce different serializations
        assert ser1 != ser2

    def test_parse_example(self):
        """Test parsing example from TFRecord format."""
        spectrogram = tf.constant(np.arange(8).reshape(2, 4), dtype=tf.float32)
        labels = tf.constant(np.arange(4).reshape(2, 2), dtype=tf.float32)
        dataset_shape = {"spectrogram": [2, 4], "labels": [2, 2]}

        serialized = _serialize_example(spectrogram, labels)
        parsed_spec, parsed_labels = _parse_example(serialized, dataset_shape)

        np.testing.assert_array_almost_equal(parsed_spec.numpy(), spectrogram.numpy())
        np.testing.assert_array_almost_equal(parsed_labels.numpy(), labels.numpy())


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_dataloader_init(self, snippet_table_fixture):
        """Test DataLoader initialization."""
        loader = DataLoader(snippet_table_fixture, n_filters=2, shuffle=False)

        assert loader.n_filters == 2
        assert loader.shuffle is False
        assert len(loader) == 2

    def test_dataloader_len(self, snippet_table_fixture):
        """Test DataLoader length method."""
        loader = DataLoader(snippet_table_fixture, n_filters=2, shuffle=False)

        assert len(loader) == len(snippet_table_fixture)

    def test_dataloader_shuffle(self, snippet_table_fixture):
        """Test DataLoader shuffle functionality."""
        loader_shuffle = DataLoader(
            snippet_table_fixture,
            n_filters=2,
            shuffle=True,
            rng=np.random.default_rng(42),
        )
        loader_no_shuffle = DataLoader(
            snippet_table_fixture, n_filters=2, shuffle=False
        )

        # Shuffled should have same content but possibly different order
        assert len(loader_shuffle) == len(loader_no_shuffle)

    def test_dataloader_iter(self, snippet_table_fixture):
        """Test DataLoader iteration."""
        loader = DataLoader(snippet_table_fixture, n_filters=2, shuffle=False)

        count = 0
        for spec, labels in loader:
            count += 1
            assert isinstance(spec, tf.Tensor)
            assert isinstance(labels, tf.Tensor)

        assert count == len(snippet_table_fixture)

    def test_dataloader_getitem(self, snippet_table_fixture):
        """Test DataLoader __getitem__ method."""
        loader = DataLoader(snippet_table_fixture, n_filters=2, shuffle=False)

        spec, labels = loader[0]

        assert isinstance(spec, tf.Tensor)
        assert isinstance(labels, tf.Tensor)
        # Spec should have added channel dimension
        assert spec.shape[-1] == 1

    def test_dataloader_reshape_labels(self, snippet_table_fixture):
        """Test label reshaping in DataLoader."""
        loader = DataLoader(snippet_table_fixture, n_filters=1, shuffle=False)

        labels = tf.constant(np.ones((4, 7)), dtype=tf.float32)
        reshaped = loader.reshape_labels(labels)

        # With n_filters=1, downsample by 2**1 = 2
        assert reshaped.shape[0] == labels.shape[0] // 2
        assert reshaped.shape[1] == labels.shape[1]

    def test_dataloader_reshape_labels_invalid(self, snippet_table_fixture):
        """Test reshape_labels raises error for invalid dimensions."""
        loader = DataLoader(snippet_table_fixture, n_filters=2, shuffle=False)

        # 5 is not divisible by 2**2=4
        labels = tf.constant(np.ones((5, 7)), dtype=tf.float32)

        with pytest.raises(ValueError):
            loader.reshape_labels(labels)

    def test_dataloader_from_csv(self, tmp_path, snippet_table_fixture):
        """Test DataLoader.from_csv class method."""
        csv_path = tmp_path / "snippet_table.csv"
        snippet_table_fixture.to_csv(csv_path, index=False)

        loader = DataLoader.from_csv(csv_path, n_filters=2, shuffle=False)

        assert len(loader) == len(snippet_table_fixture)
        assert loader.n_filters == 2


class TestDatasetSaveLoad:
    """Tests for dataset save/load functions."""

    def test_save_dataset_structure(self, tmp_path):
        """Test save_dataset creates correct structure."""
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.random.normal((10, 128, 64, 1)), tf.random.uniform((10, 64, 7)))
        )

        output_dir = tmp_path / "dataset"
        save_dataset(dataset, output_dir, examples_per_shard=5)

        assert output_dir.exists()
        assert (output_dir / "dataset_shapes.json").exists()
        # Should have 2 shards with 5 examples each
        assert (output_dir / "data_00000.tfrecord").exists()
        assert (output_dir / "data_00001.tfrecord").exists()

    def test_save_dataset_shapes_json(self, tmp_path):
        """Test that save_dataset writes correct shapes to JSON."""
        spec_shape = (128, 64, 1)
        label_shape = (64, 7)
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.random.normal((5, *spec_shape)),
                tf.random.uniform((5, *label_shape)),
            )
        )

        output_dir = tmp_path / "dataset"
        save_dataset(dataset, output_dir, examples_per_shard=10)

        with open(output_dir / "dataset_shapes.json") as f:
            shapes = json.load(f)

        assert shapes["spectrogram"] == [128, 64, 1]
        assert shapes["labels"] == [64, 7]

    def test_save_dataset_overwrite_false(self, tmp_path):
        """Test that save_dataset raises error when overwrite=False."""
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.random.normal((2, 128, 64, 1)), tf.random.uniform((2, 64, 7)))
        )

        output_dir = tmp_path / "dataset"
        output_dir.mkdir()
        (output_dir / "dummy.txt").touch()

        with pytest.raises(FileExistsError):
            save_dataset(dataset, output_dir, overwrite=False)

    def test_save_dataset_overwrite_true(self, tmp_path):
        """Test that save_dataset succeeds with overwrite=True."""
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.random.normal((2, 128, 64, 1)), tf.random.uniform((2, 64, 7)))
        )

        output_dir = tmp_path / "dataset"
        output_dir.mkdir()

        # Should not raise
        save_dataset(dataset, output_dir, overwrite=True)
        assert (output_dir / "dataset_shapes.json").exists()

    def test_save_and_load_dataset(self, tmp_path):
        """Test round-trip save and load of dataset."""
        # Create a small dataset with correct unbatched input
        spec = tf.random.normal((4, 128, 64, 1))
        labels = tf.random.uniform((4, 64, 7))
        dataset = tf.data.Dataset.from_tensor_slices((spec, labels))

        # Save dataset
        output_dir = tmp_path / "dataset"
        save_dataset(dataset, output_dir, examples_per_shard=10, dataset_length=4)

        # Load dataset
        loaded = load_dataset(output_dir, batch_size=2)

        # Check that loaded dataset has data
        count = 0
        for loaded_spec, loaded_labels in loaded:
            count += 1
            assert loaded_spec.shape[0] == 2
            assert loaded_labels.shape[0] == 2

        assert count > 0


class TestAnnotationIO:
    """Tests for annotation file reading."""

    def test_read_annotation_file(self, annotation_file_fixture):
        """Test reading annotation file."""
        df = read_annotation_file(annotation_file_fixture)

        assert len(df) == 3
        assert "recording" in df.columns
        assert "start" in df.columns
        assert "stop" in df.columns
        assert "origlabel" in df.columns

    def test_read_annotation_file_values(self, annotation_file_fixture):
        """Test annotation file content."""
        df = read_annotation_file(annotation_file_fixture)

        assert df.iloc[0]["start"] == 0.5
        assert df.iloc[0]["stop"] == 1.5
        assert df.iloc[0]["origlabel"] == "BR"
        assert df.iloc[1]["origlabel"] == "BUZZ"
        assert df.iloc[2]["origlabel"] == "WHISTLE"

    def test_read_annotation_file_recording_column(self, annotation_file_fixture):
        """Test recording column is correctly added."""
        df = read_annotation_file(annotation_file_fixture)

        expected_name = annotation_file_fixture.stem
        assert all(df["recording"] == expected_name)

    def test_read_annotation_file_custom_colnames(self, annotation_file_fixture):
        """Test reading with custom column names."""
        df = read_annotation_file(
            annotation_file_fixture, col_names=["begin", "end", "label"]
        )

        assert "begin" in df.columns
        assert "end" in df.columns
        assert "label" in df.columns
        assert df.iloc[0]["begin"] == 0.5


class TestConversionFunctions:
    """Tests for time conversion functions."""

    def test_convert_steps_to_seconds(self):
        """Test converting time steps to seconds."""
        df = pd.DataFrame({"start": [0, 10, 20], "stop": [5, 15, 25]})
        delta_t = 0.1

        result = _convert_steps_to_seconds(df, delta_t)

        assert result.iloc[0]["start"] == 0.0
        assert result.iloc[1]["start"] == 1.0
        assert result.iloc[2]["start"] == 2.0

    def test_convert_steps_to_seconds_preserves_other_columns(self):
        """Test that _convert_steps_to_seconds preserves other columns."""
        df = pd.DataFrame(
            {
                "start": [0, 10],
                "stop": [5, 15],
                "label": ["BR", "BUZZ"],
                "confidence": [0.9, 0.8],
            }
        )
        delta_t = 0.1

        result = _convert_steps_to_seconds(df, delta_t)

        assert "label" in result.columns
        assert "confidence" in result.columns
        assert result.iloc[0]["label"] == "BR"
        assert result.iloc[1]["label"] == "BUZZ"
        assert result.iloc[0]["confidence"] == 0.9
        assert result.iloc[1]["confidence"] == 0.8


class TestPredictionSaving:
    """Tests for prediction saving functions."""

    def test_save_predictions(self, tmp_path, test_messenger):
        """Test saving predictions to file."""
        msgr, _ = test_messenger
        df = pd.DataFrame(
            {
                "start": [0, 10, 20],
                "stop": [5, 15, 25],
                "label": ["BR", "BUZZ", "WHISTLE"],
                "mean_p": [0.95, 0.87, 0.92],
                "label_source": ["model", "model", "model"],
            }
        )

        output_path = tmp_path / "predictions.txt"
        save_predictions(df, output_path, delta_t=0.1, msgr=msgr)

        assert output_path.exists()

        # Read and verify
        result = pd.read_csv(output_path, sep="\t", header=None)
        assert len(result) == 3
        assert result.iloc[0][0] == 0.0  # start converted

    def test_save_predictions_default_columns(self, tmp_path):
        """Test save_predictions with default columns."""
        df = pd.DataFrame(
            {
                "start": [0, 10],
                "stop": [5, 15],
                "label": ["BR", "BUZZ"],
                "mean_p": [0.95, 0.87],
                "label_source": ["model", "model"],
            }
        )

        output_path = tmp_path / "predictions.txt"
        save_predictions(df, output_path, delta_t=0.1)

        result = pd.read_csv(output_path, sep="\t", header=None)
        assert result.shape[1] == 5  # 5 default columns

    def test_save_predictions_custom_columns(self, tmp_path):
        """Test save_predictions with custom columns."""
        df = pd.DataFrame(
            {
                "start": [0, 10],
                "stop": [5, 15],
                "label": ["BR", "BUZZ"],
                "mean_p": [0.95, 0.87],
                "label_source": ["model", "model"],
            }
        )

        output_path = tmp_path / "predictions.txt"
        save_predictions(
            df, output_path, delta_t=0.1, columns=["start", "stop", "label"]
        )

        result = pd.read_csv(output_path, sep="\t", header=None)
        assert result.shape[1] == 3  # Only 3 columns

    def test_save_prediction_probabilities(self, tmp_path):
        """Test saving prediction probabilities."""
        predictions = np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]])
        orcai_param = {
            "model": {"filters": [32, 64]},
            "calls": ["BR", "BUZZ", "WHISTLE"],
        }
        output_path = tmp_path / "predictions.txt"
        delta_t = np.float64(0.005)

        save_prediction_probabilities(predictions, orcai_param, delta_t, output_path)

        prob_file = tmp_path / "predictions_probabilities.csv.gz"
        assert prob_file.exists()

        # Read and verify structure
        result = pd.read_csv(prob_file, index_col="time")
        assert list(result.columns) == ["BR", "BUZZ", "WHISTLE"]
        assert len(result) == 2

    def test_save_prediction_probabilities_index(self, tmp_path):
        """Test that probability indices are correct."""
        predictions = np.array([[0.1, 0.2], [0.3, 0.4]])
        orcai_param = {
            "model": {"filters": [32]},  # 2**1 = 2
            "calls": ["BR", "BUZZ"],
        }
        output_path = tmp_path / "predictions.txt"
        delta_t = np.float64(0.005)

        save_prediction_probabilities(predictions, orcai_param, delta_t, output_path)

        prob_file = tmp_path / "predictions_probabilities.csv.gz"
        result = pd.read_csv(prob_file, index_col="time")

        # time_steps_per_output_step = 2**1 = 2
        # Index should be [0*0.01*2, 1*0.01*2] = [0, 0.02]
        assert len(result) == 2
