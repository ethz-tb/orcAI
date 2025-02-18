# %%
# import
import os
import zarr
import time
import json
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np

# import local
import orcAI.auxiliary as aux
import orcAI.architectures as arch


# Loader class
class ChunkedMultiZarrDataLoader(Sequence):
    def __init__(self, dataframe, batch_size, n_filters, shuffle=True):
        """
        Data loader for extracting chunks from multiple Zarr files with reshaped labels and normalized spectrograms.

        Args:
            dataframe (pd.DataFrame): DataFrame with columns ['fnstem_path', 'row_start', 'row_stop'].
            batch_size (int): Number of samples per batch.
            n_filters (int): Number of filters for reshaping labels.
            shuffle (bool): Whether to shuffle the data at the end of each epoch.
        """
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.shuffle = shuffle

        # Preload Zarr files and JSON label names
        self.zarr_files = []
        self.label_names = []
        for idx, row in self.dataframe.iterrows():
            spec_path = os.path.join(row["fnstem_path"], "spectrogram/zarr.spc")
            lbl_path = os.path.join(row["fnstem_path"], "labels/zarr.lbl")
            if not os.path.exists(lbl_path):
                print(lbl_path)
            spectrogram = zarr.open(spec_path, mode="r")
            label = zarr.open(lbl_path, mode="r")
            self.zarr_files.append((spectrogram, label))

        # Prepare indices for batches
        self.indices = [
            (idx, row["row_start"], row["row_stop"])
            for idx, row in self.dataframe.iterrows()
        ]
        if self.shuffle:
            random.shuffle(self.indices)

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
            # start_time = time.time()
            spectrogram_chunk = spectrogram[start:stop, :]
            # spec_time = time.time()
            # print(f"Time for spec chunk: {spec_time - start_time:.2f} seconds")
            label_chunk = label[start:stop, :]
            label_time = time.time()
            # print(f"Time for label chunk: {label_time - spec_time:.2f} seconds")
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
            # reshape_time = time.time()
            # print(f"Time for reshape: {reshape_time - label_time:.2f} seconds")

            # Append processed chunks to the batch
            spectrogram_batch.append(spectrogram_chunk)
            label_batch.append(label_chunk)

        return (
            tf.stack(spectrogram_batch, axis=0),
            tf.stack(label_batch, axis=0),
        )

    # def __getitem__(self, batch_index):
    #     """
    #     Retrieve a single batch, ensuring all chunks have exactly the correct length.
    #     """

    #     batch_start = batch_index * self.batch_size
    #     batch_end = batch_start + self.batch_size
    #     batch_indices = self.indices[batch_start:batch_end]

    #     spectrogram_batch, label_batch = [], []

    #     for df_index, start, stop in batch_indices:
    #         spectrogram, label = self.zarr_files[df_index]

    #         spectrogram_chunk = spectrogram[start:stop, :]
    #         label_chunk = label[start:stop, :]

    #         # Ensure correct shape before adding
    #         if spectrogram_chunk.shape[0] != (stop - start) or label_chunk.shape[0] != (
    #             stop - start
    #         ):
    #             print(
    #                 f"Skipping chunk (1) {stop}, {start}: Expected {stop - start}, but got {spectrogram_chunk.shape[0]} spectrogram and {label_chunk.shape[0]} labels",
    #                 flush=True,
    #             )
    #             continue  # Skip this chunk if dimensions are incorrect

    #         # Expand spectrogram for CNN input
    #         spectrogram_chunk = tf.expand_dims(spectrogram_chunk, axis=-1)

    #         # Reshape labels
    #         label_chunk = self.reshape_labels(
    #             tf.convert_to_tensor(label_chunk, dtype=tf.float32)
    #         )

    #         # Final shape check (after label transformation)
    #         if label_chunk.shape[0] != (stop - start) // (2**self.n_filters):
    #             print(
    #                 f"Skipping reshaped label chunk: Expected {stop - start}, but got {label_chunk.shape[0]}",
    #                 flush=True,
    #             )
    #             continue  # Skip if label reshape caused mismatching length

    #         # Append processed chunks to the batch
    #         spectrogram_batch.append(spectrogram_chunk)
    #         label_batch.append(label_chunk)

    #     # Ensure non-empty batch before stacking
    #     if not spectrogram_batch or not label_batch:
    #         raise ValueError(
    #             f"Batch {batch_index} is empty after filtering invalid chunks."
    #         )

    #     return (
    #         tf.stack(spectrogram_batch, axis=0),
    #         tf.stack(label_batch, axis=0),
    #     )

    def on_epoch_end(self):
        """
        Shuffle indices at the end of each epoch if needed.
        """
        if self.shuffle:
            random.shuffle(self.indices)


# data generator
def data_generator(loader):
    for spectrogram_batch, label_batch in loader:
        for spectrogram, label in zip(spectrogram_batch, label_batch):
            yield spectrogram, label


# load data from csv.gz files
def load_data_from_csv(csv_list, model_dict, directories_dict, computer):
    batch_size = model_dict["batch_size"]
    n_filters = len(model_dict["filters"])
    shuffle = model_dict["shuffle"]
    df_list = []
    loader_list = []
    for i, file in enumerate(csv_list):
        df = pd.read_csv(directories_dict[computer]["root_dir_tvtdata"] + file)
        print("  - snippet file:", file, "length:", len(df))
        df_list += [df]
        if i == 0:
            spectrogram = zarr.open(
                os.path.join(df.iloc[0]["fnstem_path"], "spectrogram/zarr.spc"),
                mode="r",
            )
            spectrogram_chunk = spectrogram[
                df.iloc[0]["row_start"] : df.iloc[0]["row_stop"], :
            ]
            spectrogram_chunk_shape = spectrogram_chunk.shape
            label = zarr.open(
                os.path.join(df.iloc[0]["fnstem_path"], "labels/zarr.lbl"),
                mode="r",
            )
            label_chunk = label[df.iloc[0]["row_start"] : df.iloc[0]["row_stop"], :]
            label_chunk_shape = label_chunk.shape
        loader_list += [
            ChunkedMultiZarrDataLoader(
                df_list[i],
                batch_size=batch_size,
                n_filters=n_filters,
                shuffle=shuffle,
            )
        ]
    return loader_list, spectrogram_chunk_shape, label_chunk_shape


# create data set list
def create_dataset_list(loader_list, model_dict, spectrogram_shape, label_shape):
    dataset_list = []
    for loader in loader_list:
        dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(loader),
            output_signature=(
                tf.TensorSpec(
                    shape=(spectrogram_shape[0], spectrogram_shape[1], 1),
                    dtype=tf.float32,
                ),  # Single spectrogram shape
                tf.TensorSpec(
                    shape=(label_shape[0], label_shape[1]), dtype=tf.float32
                ),  # Single label shape
            ),
        )
        dataset = (
            dataset.shuffle(buffer_size=1000)
            .batch(
                model_dict["batch_size"], drop_remainder=True
            )  # Batch size as defined in model_dict
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        dataset_list.append(dataset)
    return dataset_list


# reload tf dataset
def reload_dataset(file_path, batch_size):
    dataset = tf.data.Dataset.load(file_path)
    dataset = (
        dataset.shuffle(buffer_size=1000)
        .batch(batch_size, drop_remainder=True)  # Batch size as defined in model_dict
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset


# save tf dataset
def save_dataset(file_path, dataset):
    import shutil
    import os

    start_time = time.time()
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    tf.data.Dataset.save(dataset, file_path)
    print(
        f"   - time saving dataset {file_path}: {time.time() - start_time:.2f} seconds"
    )

    return
