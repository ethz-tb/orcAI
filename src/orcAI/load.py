from pathlib import Path
from random import shuffle as random_shuffle
import zarr
import pandas as pd
import tensorflow as tf
from keras.utils import Sequence

from orcAI.auxiliary import Messenger


class ChunkedMultiZarrDataLoader(Sequence):
    def __init__(
        self, snippet_table, batch_size, n_filters, shuffle=True, msgr=Messenger()
    ):
        """
        Data loader for extracting chunks from multiple Zarr files with reshaped labels and normalized spectrograms.

        Args:
            snippet_table (pd.DataFrame): DataFrame with columns ['recording_data_dir', 'row_start', 'row_stop'].
            batch_size (int): Number of samples per batch.
            n_filters (int): Number of filters for reshaping labels.
            shuffle (bool): Whether to shuffle the data at the end of each epoch.
        """
        self.snippet_table = snippet_table
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.shuffle = shuffle

        # Preload Zarr files and JSON label names
        self.zarr_files = []
        for _, row in self.snippet_table.iterrows():
            spectrogram_zarr_path = Path(row["recording_data_dir"]).joinpath(
                "spectrogram", "spectrogram.zarr"
            )
            labels_zarr_path = Path(row["recording_data_dir"]).joinpath(
                "labels", "labels.zarr"
            )

            # zarr open doesn't work with try/except blocks
            if labels_zarr_path.exists():
                labels = zarr.open(labels_zarr_path, mode="r")
            else:
                msgr.error(f"File not found: {labels_zarr_path}")
                msgr.error("Wrong path? Did you create the labels?")
                raise FileNotFoundError

            # zarr open doesn't work with try/except blocks
            if spectrogram_zarr_path.exists():
                spectrogram = zarr.open(spectrogram_zarr_path, mode="r")
            else:
                msgr.error(f"File not found: {spectrogram_zarr_path}")
                msgr.error("Wrong path? Did you create the spectrogram?")
                raise FileNotFoundError

            self.zarr_files.append((spectrogram, labels))

        # Prepare indices for batches
        self.indices = [
            (idx, row["row_start"], row["row_stop"])
            for idx, row in self.snippet_table.iterrows()
        ]
        if self.shuffle:
            random_shuffle(self.indices)

    def __len__(self):
        """
        Number of batches per epoch.
        """
        return len(self.indices) // self.batch_size

    def reshape_labels(self, arr):
        """
        Reshape and process labels using the provided number of filters (n_filters) to achieve a time resolution on labels which is time_steps_spectogram//2**n_filters.
        """

        if arr.shape[0] % (2**self.n_filters) == 0:
            # Reshape the array to group rows for averaging
            new_shape = (
                arr.shape[0] // (2**self.n_filters),
                2**self.n_filters,
                arr.shape[1],
            )
            reshaped = tf.reshape(
                arr, new_shape
            )  # Shape: (time_steps_labels, downsample_factor, num_labels)
            # Compute the mean along the downsampling axis
            averaged = tf.reduce_mean(
                reshaped, axis=1
            )  # Shape: (time_steps_labels, num_labels)
            arr_out = tf.round(averaged)  # round to next integer
            return arr_out
        else:
            # print("arr.shape", arr.shape, flush=True)
            # arr_out = -1 * np.ones((46, 7))
            # return arr_out
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
            # Reshape labels
            label_chunk = self.reshape_labels(
                tf.convert_to_tensor(label_chunk, dtype=tf.float32)
            )
            if label_chunk.shape[0] == 0:
                continue

            # Append processed chunks to the batch
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
            random_shuffle(self.indices)


# data generator
def data_generator(loader):
    for spectrogram_batch, label_batch in loader:
        for spectrogram, label in zip(spectrogram_batch, label_batch):
            yield spectrogram, label


# load data from csv.gz files
def load_data_from_snippet_csv(
    csv_paths,
    model_parameter,
    msgr=Messenger(),
):
    msgr.part("Loading data from snippet csv files")
    batch_size = model_parameter["batch_size"]
    n_filters = len(model_parameter["filters"])
    shuffle = model_parameter["shuffle"]
    loader_list = {}
    for i, csv_path in enumerate(csv_paths):
        snippet_table = pd.read_csv(csv_path)
        msgr.info(f"snippet file: {csv_path.stem}, length: {len(snippet_table)}")
        if i == 0:
            spectrogram = zarr.open(
                Path(snippet_table.iloc[0]["recording_data_dir"]).joinpath(
                    "spectrogram", "spectrogram.zarr"
                ),
                mode="r",
            )
            spectrogram_chunk = spectrogram[
                snippet_table.iloc[0]["row_start"] : snippet_table.iloc[0]["row_stop"],
                :,
            ]
            label = zarr.open(
                Path(snippet_table.iloc[0]["recording_data_dir"]).joinpath(
                    "labels", "labels.zarr"
                ),
                mode="r",
            )
            label_chunk = label[
                snippet_table.iloc[0]["row_start"] : snippet_table.iloc[0]["row_stop"],
                :,
            ]
        loader_list[csv_path.with_suffix("").stem] = ChunkedMultiZarrDataLoader(
            snippet_table,
            batch_size=batch_size,
            n_filters=n_filters,
            shuffle=shuffle,
        )

    return (
        loader_list,
        spectrogram_chunk.shape,
        label_chunk.shape,
    )


# reload tf dataset
def reload_dataset(file_path, batch_size):
    dataset = tf.data.Dataset.load(str(file_path))
    dataset = (
        dataset.shuffle(buffer_size=1000)
        .batch(
            batch_size, drop_remainder=True
        )  # Batch size as defined in model_parameter
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset
