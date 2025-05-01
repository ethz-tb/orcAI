import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import zarr

from orcAI.json_encoder import JsonEncoderExt

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)
SHUFFLE_BUFFER_SIZE = 1000


class DataLoader:
    """
    Data loader for extracting snippets from multiple Zarr files with reshaped labels and normalized spectrograms.
    """

    def __init__(
        self,
        snippet_table: pd.DataFrame,
        n_filters: int,
        shuffle: bool = True,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        Parameters:
        -----------
        snippet_table: pd.DataFrame
            DataFrame with columns ['recording_data_dir', 'row_start', 'row_stop'].
        n_filters: int
            Number of filters for reshaping labels.
        shuffle: bool
            Whether to shuffle data after loading.
        rng: np.random.Generator
            Random number generator for shuffling.
        """

        if shuffle:
            self.snippet_table = snippet_table.sample(
                frac=1, axis="index", random_state=rng
            ).reset_index(drop=True)
        else:
            self.snippet_table = snippet_table

        self.indices = snippet_table.index
        self.n_filters = n_filters
        self.shuffle = shuffle
        self.rng = rng

        # Preload Zarr files and JSON label names
        self.zarr_files = [
            self._load_zarr_files(path)
            for path in self.snippet_table.recording_data_dir
        ]

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        n_filters: int,
        shuffle: bool = True,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        Create a DataLoader from a snippet table saved at CSV file.
        """
        import pandas as pd

        snippet_table = pd.read_csv(path)
        return cls(snippet_table, n_filters, shuffle, rng)

    def __len__(self):
        """
        Number of snippets
        """
        return len(self.indices)

    def __iter__(self):
        """
        Return an iterator over the data loader.
        """
        for i in self.indices:
            yield self[i]

    def _load_zarr_files(self, path: Path | str):
        """
        Load Zarr files from the provided path.
        """
        path = Path(path)
        spectrogram_zarr_path = path.joinpath("spectrogram", "spectrogram.zarr")
        labels_zarr_path = path.joinpath("labels", "labels.zarr")

        labels = zarr.open(labels_zarr_path, mode="r")
        spectrogram = zarr.open(spectrogram_zarr_path, mode="r")

        return spectrogram, labels

    def reshape_labels(self, labels):
        """
        Reshape and process labels using the provided number of filters (n_filters)
        to achieve a time resolution on labels which is time_steps_spectogram//2**n_filters.
        """

        if labels.shape[0] % (2**self.n_filters) == 0:
            # Reshape the array to group rows for averaging
            new_shape = (
                labels.shape[0] // (2**self.n_filters),
                2**self.n_filters,
                labels.shape[1],
            )
            reshaped = tf.reshape(
                labels, new_shape
            )  # Shape: (time_steps_labels, downsample_factor, num_labels)
            # Compute the mean along the downsampling axis
            averaged = tf.reduce_mean(
                reshaped, axis=1
            )  # Shape: (time_steps_labels, num_labels)
            labels_out = tf.round(averaged)  # round to next integer
            return labels_out
        else:
            raise ValueError(
                "The number of rows in 'arr' must be divisible by 2**'n_filters'."
            )

    def __getitem__(self, index: int):
        """
        Retrieve a single batch, aggregating data from multiple Zarr files if needed.
        """

        spectrogram, label = self.zarr_files[index]
        start = self.snippet_table.iloc[index]["row_start"]
        stop = self.snippet_table.iloc[index]["row_stop"]
        spectrogram_chunk = spectrogram[start:stop, :]
        label_chunk = label[start:stop, :]

        # Normalize spectrogram
        spectrogram_chunk = tf.expand_dims(spectrogram_chunk, axis=-1)

        # Reshape labels
        label_chunk = self.reshape_labels(
            tf.convert_to_tensor(label_chunk, dtype=tf.float32)
        )

        return (spectrogram_chunk, label_chunk)


def load_dataset(
    path: Path | str,
    batch_size: int,
    compression: str = "GZIP",
    seed: int | list[int] = None,
):
    dataset = (
        tf.data.Dataset.load(str(path), compression=compression)
        .shuffle(
            buffer_size=SHUFFLE_BUFFER_SIZE,
            seed=int(np.random.SeedSequence(seed).generate_state(1)[0]),
        )
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return dataset


def save_dataset(
    dataset: tf.data.Dataset,
    path: Path | str,
    overwrite: bool = False,
    compression: str = "GZIP",
) -> None:
    """
    Save a dataset
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"File {path} already exists.")
    dataset.save(str(path), compression=compression)
    return


def write_vector_to_json(vector, filename):
    """write out equally spaced vector in short form with min, max and length"""
    dictionary = {"min": vector[0], "max": vector[-1], "length": len(vector)}
    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=4)
    return


def generate_times_from_spectrogram(filename):
    """read and generate equally spaced vector in short form from min, max and length"""
    with open(filename, "r") as f:
        dictionary = json.load(f)
    return np.linspace(dictionary["min"], dictionary["max"], dictionary["length"])


def read_json(filename):
    """Read a JSON file into a dictionary"""
    with open(filename, "r") as file:
        dictionary = json.load(file)
    return dictionary


def write_json(dictionary, filename):
    """write dictionary into json file"""
    json_string = json.dumps(dictionary, indent=4, cls=JsonEncoderExt)
    with open(filename, "w") as file:
        file.write(json_string)
    return


def save_as_zarr(
    obj,
    filename: Path,
    compressors: dict[str, list] = {
        "bytes": [{"configuration": {}, "name": "gzip"}],
        "numeric": [{"configuration": {}, "name": "gzip"}],
        "string": [{"configuration": {}, "name": "gzip"}],
    },
):
    """write object to zarr file"""

    zarr.config.set({"array.v3_default_compressors": compressors})

    zarr_file = zarr.open(
        filename,
        mode="w",
        shape=obj.shape,
        chunks=(2000, obj.shape[1]),
        dtype="float32",
    )

    zarr_file[:] = obj
    return


def read_annotation_file(annotation_file_path):
    """read annotation file and return with recording as additional column"""
    annotation_file = pd.read_csv(
        annotation_file_path,
        sep="\t",
        encoding="utf-8",
        header=None,
        names=["start", "stop", "origlabel"],
    )
    annotation_file["recording"] = Path(annotation_file_path).stem
    return annotation_file[["recording", "start", "stop", "origlabel"]]


def load_orcai_model(model_dir: Path) -> tuple[keras.Model, dict, dict]:
    import keras

    from orcAI.architectures import (
        MaskedBinaryAccuracy,
        MaskedBinaryCrossentropy,
        res_net_LSTM_arch,
    )

    orcai_parameter = read_json(model_dir.joinpath("orcai_parameter.json"))
    shape = read_json(model_dir.joinpath("model_shape.json"))

    if model_dir.joinpath(orcai_parameter["name"] + ".keras").exists():
        model = keras.saving.load_model(
            model_dir.joinpath(orcai_parameter["name"] + ".keras"),
            custom_objects=None,
            compile=True,
            safe_mode=True,
        )
    elif model_dir.joinpath("model_weights.h5").exists():
        # legacy model
        model = res_net_LSTM_arch(**shape, **orcai_parameter["model"])
        model.load_weights(model_dir.joinpath("model_weights.h5"))

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=orcai_parameter["model"]["learning_rate"]
            ),
            loss=MaskedBinaryCrossentropy(),
            metrics=[MaskedBinaryAccuracy()],
        )
    else:
        raise ValueError(
            f"Couldn't find model weights (model_weights.h5) or keras model file in {model_dir}"
        )

    return model, orcai_parameter, shape
