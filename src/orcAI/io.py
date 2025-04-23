import json
from pathlib import Path

import tf_keras as keras
import numpy as np
import pandas as pd
import tensorflow as tf
import zarr

from orcAI.json_encoder import JsonEncoderExt

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)
SHUFFLE_BUFFER_SIZE = 1000


class DataLoader(keras.utils.Sequence):
    """
    Data loader for extracting snippets from multiple Zarr files with reshaped labels and normalized spectrograms.
    """

    def __init__(
        self,
        snippet_table: pd.DataFrame,
        batch_size: int,
        n_filters: int,
        shuffle: bool = True,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        Parameters:
        -----------
        snippet_table: pd.DataFrame
            DataFrame with columns ['recording_data_dir', 'row_start', 'row_stop'].
        batch_size: int
            Size of each batch.
        n_filters: int
            Number of filters for reshaping labels.
        shuffle: bool
            Whether to shuffle data after loading.
        rng: np.random.Generator
            Random number generator for shuffling.
        """
        self.snippet_table = snippet_table
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.shuffle = shuffle
        self.rng = rng

        self.zarr_files = [
            self._load_zarr_files(path)
            for path in self.snippet_table.recording_data_dir
        ]

        # Prepare indices for batches
        self.indices = [
            (idx, row["row_start"], row["row_stop"])
            for idx, row in self.snippet_table.iterrows()
        ]

        if self.shuffle:
            rng.shuffle(self.indices)

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        **kwargs,
    ):
        """
        Create a DataLoader from a snippet table saved at path.
        """
        import pandas as pd

        snippet_table = pd.read_csv(path)
        return cls(snippet_table, **kwargs)

    def __len__(self):
        """
        Number of batches per epoch.
        """
        return len(self.indices) // self.batch_size

    def reshape_labels(self, labels):
        """
        Reshape and process labels using the provided number of filters (n_filters) to achieve a time resolution on labels which is time_steps_spectogram//2**n_filters.
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

    def __getitem__(self, batch_index):
        """
        Retrieve a single batch, aggregating data from multiple Zarr files if needed.
        """

        batch_start = batch_index * self.batch_size
        batch_end = batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]

        spectrogram_batch, label_batch = [], []

        for df_index, start, stop in batch_indices:
            spectrogram, label = self.zarr_files[df_index]
            spectrogram_chunk = spectrogram[start:stop, :]
            label_chunk = label[start:stop, :]
            if spectrogram_chunk.shape[0] == 0:
                continue
            spectrogram_chunk = tf.expand_dims(spectrogram_chunk, axis=-1)

            if label_chunk.shape[0] != 736:
                print(spectrogram_chunk.shape[0], label_chunk.shape[0], flush=True)

            label_chunk = self.reshape_labels(
                tf.convert_to_tensor(label_chunk, dtype=tf.float32)
            )
            if label_chunk.shape[0] == 0:
                continue

            spectrogram_batch.append(spectrogram_chunk)
            label_batch.append(label_chunk)

        return (
            tf.stack(spectrogram_batch, axis=0),
            tf.stack(label_batch, axis=0),
        )

    def on_epoch_end(self):
        """
        Shuffle indices at the end of each epoch if needed.
        """
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _load_zarr_files(self, path: Path):
        """
        Load Zarr files from the provided path.
        """
        path = Path(path)
        spectrogram_zarr_path = path.joinpath("spectrogram", "spectrogram.zarr")
        labels_zarr_path = path.joinpath("labels", "labels.zarr")

        labels = zarr.open(labels_zarr_path, mode="r")
        spectrogram = zarr.open(spectrogram_zarr_path, mode="r")

        return spectrogram, labels


# data generator
def data_generator(loader):
    for spectrogram_batch, label_batch in loader:
        for spectrogram, label in zip(spectrogram_batch, label_batch):
            yield spectrogram, label


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
    import tf_keras as keras

    from orcAI.architectures import (
        MaskedAUC,
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
            optimizer="adam",
            loss=MaskedBinaryCrossentropy(),
            metrics=[MaskedAUC(), MaskedBinaryAccuracy()],
        )
    else:
        ValueError(
            f"Couldn't find model weights (model_weights.h5) or keras model file in {model_dir}"
        )

    return model, orcai_parameter, shape
