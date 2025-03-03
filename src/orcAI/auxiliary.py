import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import click
import zarr


class JsonEncoderExt(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class Messenger:
    """Class for printing messages with different levels of verbosity and indentation"""

    def __init__(self, n_indent=0, verbosity=2, indent_str="    ", file=None):
        """
        Initialize the Messenger object with the specified verbosity level and indentation.

        Parameter:
        n_indent (int): The initial indentation level.
        verbosity (int): The verbosity level.
        file (file): The file to write the messages to.
        indent_str (str): The string to use for indentation.
        """
        self.n_indent = n_indent
        self.verbosity = verbosity
        self.file = file
        self.indent_str = indent_str

    def print(
        self, message, indent=0, set_indent=None, prepend="", severity=2, **kwargs
    ):
        """
        Print a message with the specified indentation level and verbosity.

        Parameters:
        message (str | dict | list): The message (or dict or list) to print.
        indent (int): The number of additional indent levels after this message.
        set_indent (int): The absolute indent level for this message.
        prepend (str): A string to prepend to the message.
        severity (int): The severity level of the message (0: error, 1: warning, 2: info, ...).
        **kwargs: Additional keyword arguments for click.style.
        """
        if self.verbosity < severity:
            return
        if set_indent is not None:
            self.n_indent = set_indent

        if isinstance(message, dict) or isinstance(message, list):
            message = self.dict_to_str(message)
        elif isinstance(message, pd.DataFrame | pd.Series):
            message = self.pd_to_str(message)
        else:
            message = self.indent_str * self.n_indent + prepend + message

        message = click.style(message, **kwargs)

        click.echo(message, file=self.file)

        self.n_indent = self.n_indent + indent

    def info(self, message, indent=0, set_indent=None, severity=2, **kwargs):
        """Print a message."""
        self.print(message, indent, set_indent, severity=severity, **kwargs)

    def part(self, message, indent=1, set_indent=0, severity=1, **kwargs):
        """Print a message in bold at indent 0 to indicate a new part"""
        self.print(
            message,
            indent,
            set_indent,
            prepend="🐳 ",
            severity=severity,
            bold=True,
            **kwargs,
        )

    def success(self, message, indent=0, set_indent=0, severity=1, **kwargs):
        """Print a success message."""
        self.print(
            message,
            indent,
            set_indent,
            prepend="🐳 ",
            severity=severity,
            fg="green",
            **kwargs,
        )

    def warning(self, message, indent=0, set_indent=None, severity=1, **kwargs):
        """Print a warning message."""
        self.print(
            message,
            indent,
            set_indent,
            prepend="‼️ ",
            severity=severity,
            fg="yellow",
            **kwargs,
        )

    def error(self, message, indent=0, set_indent=None, severity=0, **kwargs):
        """Print an error message."""
        self.print(
            message,
            indent,
            set_indent,
            prepend="❌ ",
            severity=severity,
            fg="red",
            **kwargs,
        )

    def print_memory_usage(self, indent=0, set_indent=None, severity=2, **kwargs):
        """print memory usage"""
        if self.verbosity < severity:
            return
        from psutil import Process

        process = Process(os.getpid())
        self.info(
            f"memory usage: {process.memory_info().rss / 1024 ** 2} MB",
            indent=indent,
            set_indent=set_indent,
            severity=severity,
            **kwargs,
        )

    def print_directory_size(
        self, directory, indent=0, set_indent=None, severity=2, **kwargs
    ):
        """print size of directory"""

        if self.verbosity < severity:
            return
        from humanize import naturalsize

        total_size = sum(
            f.stat().st_size for f in Path(directory).rglob("*") if f.is_file()
        )
        self.info(
            f"Size on disk of {Path(directory).stem}: {naturalsize(total_size, format='%.2f')}",
            indent=indent,
            set_indent=set_indent,
            severity=severity,
            **kwargs,
        )

    def confusion_matrices_to_str(self, confusion_matrices):
        """Convert confusion matrices to a formatted string."""
        message = ""
        for label, cm in confusion_matrices.items():
            message = message.append(
                "\n".join(
                    [
                        f"............................",
                        f"Label: {label}, total={cm['Total']}",
                        f"   Predicted:     | POS     | NEG  ",
                        f"   Actual:    POS | {100*cm['TP']:.5f} | {100*cm['FN']:.5f} ",
                        f"              -------------------------- ",
                        f"   Actual:    NEG | {100*cm['FP']:.5f} | {100*cm['TN']:.5f} ",
                    ]
                )
            )
        return message

    def dict_to_str(self, dictionary):
        """Convert a dictionary or list to a formatted string with indentation."""
        json_string = json.dumps(dictionary, indent=4, cls=JsonEncoderExt)
        indented_json = "\n".join(
            self.indent_str * self.n_indent + line for line in json_string.splitlines()
        )
        return indented_json

    def pd_to_str(self, obj):
        """Convert a DataFrame to a formatted string with indentation."""
        obj_string = obj.to_string()
        indented_obj = "\n".join(
            self.indent_str * self.n_indent + line for line in obj_string.splitlines()
        )
        return indented_obj


def write_vector_to_json(vector, filename):
    """write out equally spaced vector in short form with min, max and length"""
    dictionary = {"min": vector[0], "max": vector[-1], "length": len(vector)}
    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=4)
    return


def read_json_to_vector(filename):
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


def get_all_files_with_ext(directory, extension):
    """
    Recursively get all ".ext" files from the specified directory.

    Args:
        directory (str): The root directory to search in.

    Returns:
        list: A list of full file paths to all files with extension found.
    """
    all_files = []

    # Walk through the directory recursively
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension):
                all_files.append(os.path.join(root, file))

    return all_files


def resolve_file_paths(
    directory, file_names, extension="", msgr=Messenger(verbosity=2)
):
    """
    Resolve file paths for a list of file names in a directory.

    Args:
        directory (str): The root directory to search in.
        file_names (list): A list of file names to resolve.
        extension (str): The file extension to append to the file names.

    Returns:
        list: A list of full file paths for the specified file names.
    """
    file_paths = []
    if extension[0] != ".":
        extension = "." + extension
    for file_name in file_names:
        results = list(Path(directory).rglob(file_name + extension))
        if len(results) == 0:
            msgr.warning(f"No {extension} file found for {file_name}. Returning None.")
            results = [None]
        elif len(results) > 1:
            msgr.warning(f"WARNING: Multiple files found for {file_name}", indent=1)
            msgr.info(results)
            msgr.warning(f"Returning the first path.", indent=-1)
        file_paths.append(results[0])
    return file_paths


def resolve_recording_data_dir(recording, recording_data_dir):
    if Path(recording_data_dir, recording).exists():
        return Path(recording_data_dir, recording)
    else:
        return None


def filter_recordings(files, eliminate, msgr=Messenger(verbosity=2)):
    """remove filenames containing patterns in list eliminate"""
    msgr.info(f"Filtering {len(files)} files...")
    for e in eliminate:
        files = [f for f in files if not e in f.name]
        msgr.info(
            f"Remaining files after filtering files that contain {e}: {len(files)}"
        )
    return files


def compute_confusion_matrix(y_true_batch, y_pred_batch, label_names, mask_value=-1):
    """
    Compute the confusion matrix for each label across the entire batch.

    Args:
        y_true_batch (np.ndarray): Ground truth binary labels with shape (batch_size, time_steps, num_labels).
        y_pred_batch (np.ndarray): Predicted  labels with shape (batch_size, time_steps, num_labels).
        mask_value (int, optional): Mask value in y_true_batch that indicates missing labels. Defaults to -1.

    Returns:
        dict: A dictionary where keys are label indices and values are confusion matrices (2x2 numpy arrays).
    """
    # Ensure inputs are numpy arrays
    y_true_batch = np.array(y_true_batch)
    y_pred_binary_batch = (y_pred_batch >= 0.5).astype(int)
    y_pred_binary_batch = np.array(y_pred_binary_batch)

    # Validate input shapes
    assert (
        y_true_batch.shape == y_pred_binary_batch.shape
    ), "Shapes of y_true_batch and y_pred_binary_batch must match"

    # Extract the number of labels
    num_labels = y_true_batch.shape[-1]

    # Initialize a dictionary to store confusion matrices for each label
    confusion_matrices = {}

    for label_idx in range(len(label_names)):
        # Flatten the predictions and ground truth for the current label
        y_true_flat = y_true_batch[:, :, label_idx].flatten()
        y_pred_flat = y_pred_binary_batch[:, :, label_idx].flatten()

        # Apply the mask to exclude masked values
        mask = y_true_flat != mask_value
        y_true_filtered = y_true_flat[mask]
        y_pred_filtered = y_pred_flat[mask]

        # Compute the confusion matrix for the current label
        [tn, fp], [fn, tp] = confusion_matrix(
            y_true_filtered, y_pred_filtered, labels=[0, 1]
        )
        tot = tn + fp + fn + tp
        cm = {
            "TP": float(tp / tot),
            "FN": float(fn / tot),
            "FP": float(fp / tot),
            "TN": float(tn / tot),
            "Total": int(tot),
        }
        # Store the confusion matrix
        confusion_matrices[label_names[label_idx]] = cm

    return confusion_matrices


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def find_consecutive_ones(binary_vector):
    """
    Finds the start and end indices of consecutive sequences of ones in a binary vector.

    Args:
        binary_vector (np.ndarray): A binary vector (1D array of 0s and 1s).

    Returns:
        List[Tuple[int, int]]: A list of (start, end) indices for each sequence of ones.
    """
    # Find where the binary vector changes
    diff = np.diff(binary_vector, prepend=0, append=0)

    # Start indices are where 0 → 1, end indices are where 1 → 0
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0] - 1  # Adjust to include the last 1

    # Combine starts and ends into a list of tuples
    return starts, stops


def recording_table_show_func(index, recording_table):
    """Show the recording and channel for a given index in the recording table. Used for progress bars."""
    if index is not None:
        return (
            recording_table.loc[index, "recording"]
            + ", Ch "
            + str(recording_table.loc[index, "channel"])
        )
