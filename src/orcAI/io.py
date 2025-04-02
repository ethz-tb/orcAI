from pathlib import Path
import zarr
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import time

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)

from orcAI.auxiliary import Messenger


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

        # zarr open doesn't work with try/except blocks
        if labels_zarr_path.exists():
            labels = zarr.open(labels_zarr_path, mode="r")
        else:
            raise FileNotFoundError(f"File not found: {labels_zarr_path}")

        # zarr open doesn't work with try/except blocks
        if spectrogram_zarr_path.exists():
            spectrogram = zarr.open(spectrogram_zarr_path, mode="r")
        else:
            raise FileNotFoundError(f"File not found: {spectrogram_zarr_path}")
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


def serialize_example(spectrogram, labels):
    feature_dict = {
        "spectrogram": tf.train.Feature(
            float_list=tf.train.FloatList(value=spectrogram.numpy().flatten())
        ),
        "labels": tf.train.Feature(
            float_list=tf.train.FloatList(value=labels.numpy().flatten())
        ),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


def parse_example(
    proto,
    dataset_shape: dict["spectrogram" : tuple[int, int, int], "labels":[int, int]],
):
    feature_description = {
        "spectrogram": tf.io.FixedLenFeature(
            np.prod(dataset_shape["spectrogram"]), tf.float32
        ),
        "labels": tf.io.FixedLenFeature(np.prod(dataset_shape["labels"]), tf.float32),
    }
    example = tf.io.parse_single_example(proto, feature_description)

    spectrogram = tf.reshape(example["spectrogram"], dataset_shape["spectrogram"])
    labels = tf.reshape(example["labels"], dataset_shape["labels"])

    return spectrogram, labels


def _load_dataset(
    file_path: str,
    dataset_shape: dict["spectrogram" : tuple[int, int, int], "labels":[int, int]],
):
    """
    Load a dataset from a TFRecord file. (seperate function mainly for testing)
    """
    dataset = tf.data.TFRecordDataset(
        str(file_path), num_parallel_reads=8, compression_type="GZIP"
    ).map(lambda proto: parse_example(proto, dataset_shape))
    return dataset


# reload tf dataset
def load_dataset(
    file_path: str,
    dataset_shape: dict["spectrogram" : tuple[int, int, int], "labels":[int, int]],
    batch_size: int,
    n_batches: int = None,
    seed: int | list[int] = None,
):
    dataset = (
        _load_dataset(file_path, dataset_shape)
        .shuffle(
            buffer_size=1000,
            seed=int(np.random.SeedSequence(seed).generate_state(1)[0]),
        )
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    if n_batches is not None:
        dataset = dataset.take(n_batches)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(n_batches))
    return dataset


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
    json_string = json.dumps(dictionary, indent=4)
    with open(filename, "w") as file:
        file.write(json_string)
    return


def save_as_zarr(obj, filename, msgr=Messenger(verbosity=2)):
    """write object to zarr file"""
    start_time = time.time()
    zarr_file = zarr.open(
        filename,
        mode="w",
        shape=obj.shape,
        chunks=(2000, obj.shape[1]),
        dtype="float32",
        compressor=zarr.Blosc(cname="zlib"),
    )
    zarr_file[:] = obj
    save_time = time.time()
    msgr.info(f"Time for for saving to disk: {save_time - start_time:.2f} seconds")
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
