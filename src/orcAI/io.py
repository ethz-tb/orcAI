import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import zarr

from orcAI.auxiliary import Messenger
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
        Parameter:
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
) -> tf.data.Dataset:
    """Load a tf.data.Dataset from a directory.

    Parameter
    ----------
    path : Path | str
        Path to the directory containing the dataset.
    batch_size : int
        Batch size for the dataset.
    compression : str
        Compression type for the dataset. Default is "GZIP".
    seed : int | list[int]
        Random seed for shuffling the dataset. Default is None.

    Returns
    -------
    dataset: tf.data.Dataset
        The loaded dataset.
    """
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
    Save a tf.data.Dataset
    Parameter
    ----------
    dataset : tf.data.Dataset
        The dataset to save.
    path : Path | str
        Path to the directory where the dataset will be saved.
    overwrite : bool
        If True, overwrite the existing dataset. Default is False.
    compression : str
        Compression type for the dataset. Default is "GZIP".

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If the dataset already exists and overwrite is False.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"File {path} already exists.")
    dataset.save(str(path), compression=compression)
    return


def write_vector_to_json(vector: list, filename: Path | str) -> None:
    """Write out equally spaced vector in short form with min, max and length

    Parameter
    ----------
    vector : np.ndarray
        The vector to write to the JSON file.
    filename : str
        The name of the JSON file to write to.

    Returns
    -------
    None
    """
    dictionary = {"min": vector[0], "max": vector[-1], "length": len(vector)}
    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=4)
    return


def generate_times_from_spectrogram(filename: Path | str) -> np.ndarray:
    """Read and generate equally spaced vector in short form from min, max and length

    Parameter
    ----------
    filename : str
        The name of the JSON file to read from.

    Returns
    -------
    np.ndarray
        The generated equally spaced vector.
    """
    with open(filename, "r") as f:
        dictionary = json.load(f)
    return np.linspace(dictionary["min"], dictionary["max"], dictionary["length"])


def read_json(filename: Path | str) -> dict:
    """Read a JSON file into a dictionary

    Parameter
    ----------
    filename : str
        The name of the JSON file to read from.

    Returns
    -------
    dict
        The dictionary containing the JSON data.
    """
    with open(filename, "r") as file:
        dictionary = json.load(file)
    return dictionary


def write_json(dictionary, filename) -> None:
    """write dictionary into json file
    Parameter
    ----------
    dictionary : dict
        The dictionary to write to the JSON file.
    filename : str
        The name of the JSON file to write to.

    Returns
    -------
    None
    """
    json_string = json.dumps(dictionary, indent=4, cls=JsonEncoderExt)
    with open(filename, "w") as file:
        file.write(json_string)
    return


def save_as_zarr(
    obj: any,
    filename: Path,
    compressors: dict[str, list] = {
        "bytes": [{"configuration": {}, "name": "gzip"}],
        "numeric": [{"configuration": {}, "name": "gzip"}],
        "string": [{"configuration": {}, "name": "gzip"}],
    },
) -> None:
    """write object to zarr file
    Parameter
    ----------
    obj : any
        The object to write to the Zarr file.
    filename : str
        The name of the Zarr file to write to.
    compressors : dict
        The compressors to use for the Zarr file. Default is GZIP.

    Returns
    -------
    None
    """

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


def read_annotation_file(annotation_file_path) -> pd.DataFrame:
    """read annotation file and return with recording as additional column
    Parameter
    ----------
    annotation_file_path : str
        The path to the annotation file.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the annotation data with an additional column for the recording name.
    """
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
    """Load a trained orcAI model from a directory.

    Parameter
    ----------
    model_dir : Path
        Path to the directory containing the model files.

    Returns
    -------
    tuple[keras.Model, dict, dict]
        A tuple containing the loaded model, orcAI parameter, and model shape.

    Raises
    ------
    ValueError
        If the model weights or keras model file is not found in the specified directory.
    """
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


def _convert_times_to_seconds(
    predicted_labels: pd.DataFrame,
    delta_t: float,
) -> pd.DataFrame:
    """
    Converts the start and stop times of predicted labels from time steps to seconds.
    Parameters
    ----------
    predicted_labels : pd.DataFrame
        DataFrame with predicted labels containing 'start' and 'stop' columns in time steps.
    delta_t : float
        Time step duration in seconds.
    time_steps_per_output_step : int
        Number of time steps per output step.
    Returns
    -------
    pd.DataFrame
        DataFrame with 'start' and 'stop' columns converted to seconds.
    """
    predicted_labels.loc[:, "start"] = predicted_labels.loc[:, "start"] * delta_t
    predicted_labels.loc[:, "stop"] = predicted_labels.loc[:, "stop"] * delta_t
    return predicted_labels


def save_predictions(
    predicted_labels: pd.DataFrame,
    output_path: Path | str,
    delta_t: float,
    msgr: Messenger = Messenger(verbosity=0),
) -> None:
    """
    Saves the predicted labels to a file.

    Parameters
    ----------
    predicted_labels : pd.DataFrame
        DataFrame with predicted labels containing 'start', 'stop', and 'label' columns.
    output_path : Path | str
        Path to the output file.
    delta_t : float
        Time step duration in seconds.
    msgr : Messenger
        Messenger object for logging.
    """
    predicted_labels = _convert_times_to_seconds(predicted_labels, delta_t)
    predicted_labels[["start", "stop", "label"]].round(4).to_csv(
        output_path, sep="\t", index=False, header=False
    )
    msgr.info(f"Predictions saved to {output_path}")
    return


def save_prediction_probabilities(
    aggregated_predictions: np.ndarray,
    orcai_parameter: dict,
    delta_t: float,
    output_path: Path | str,
    msgr: Messenger = Messenger(verbosity=0),
) -> None:
    """
    Saves the prediction probabilities to a file.
    Parameters
    ----------
    aggregated_predictions : np.ndarray
        Array with aggregated predictions.
    orcai_parameter : dict
        orcAI parameter dictionary.
    delta_t : float
        Time step duration in seconds.
    output_path : Path | str
        Path to the output file.
    msgr : Messenger
        Messenger object for logging.
    """
    predictions_path = output_path.with_name(f"{output_path.stem}_probabilities.csv.gz")
    pd.DataFrame(
        aggregated_predictions,
        columns=orcai_parameter["calls"],
        index=delta_t * range(len(aggregated_predictions)),
    ).to_csv(predictions_path, index_label="time", compression="gzip")
    msgr.info(f"Prediction probabilities saved to {predictions_path}")
    return
