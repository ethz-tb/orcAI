import json
import os
import time
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import click
import numpy as np
import pandas as pd
from humanize import naturalsize

from orcAI.json_encoder import JsonEncoderExt

# Sees IDs for different parts of the pipeline
SEED_ID_MAKE_SNIPPET_TABLE = 1
SEED_ID_FILTER_SNIPPET_TABLE = 2
SEED_ID_CREATE_DATALOADER = {"train": 3, "val": 4, "test": 5, "unfiltered_test": 6}
SEED_ID_LOAD_TRAIN_DATA = 7
SEED_ID_LOAD_VAL_DATA = 8
SEED_ID_LOAD_TEST_DATA = 9
SEED_ID_UNFILTERED_TEST_DATA = 10

# Value used to mask labels in the dataset
MASK_VALUE = -1.0


class Messenger:
    """Class for printing messages with different levels of verbosity and indentation"""

    def __init__(
        self,
        title: str = None,
        n_indent: int = 0,
        verbosity: int = 2,
        indent_str: str = "    ",
        show_part_times: bool = True,
        file: Path = None,
    ):
        """
        Initialize the Messenger object with the specified verbosity level and indentation.

        Parameter
        ---------
        title: str
            The title to print at the start.
        n_indent: int
            The initial indentation level.
        verbosity: int
            The verbosity level (0: error, 1: warning, 2: info, 3:Debug).
        indent_str: str
            The string to use for indentation.
        show_part_times: bool
            Whether to show the time taken for each part.
        file: Path
            The file to write the messages to (default: None, which means stdout).
        """
        self.n_indent = n_indent
        self.verbosity = verbosity
        self.file = file
        self.indent_str = indent_str
        self.show_part_times = show_part_times
        self.start_time = time.time()
        self.part_times = []
        if title is not None:
            self.start(title, severity=2)

    def print(
        self,
        message: str,
        indent: int = 0,
        set_indent: int | None = None,
        prepend: str = "",
        severity: int = 2,
        **kwargs,
    ):
        """
        Print a message with the specified indentation level and verbosity.

        Parameter
        ---------
        message: str | dict | list
            The message (or dict or list) to print.
        indent: int
            The number of additional indent levels after this message.
        set_indent: int
            The absolute indent level for this message.
        prepend: str
            A string to prepend to the message.
        severity: int
            The severity level of the message (0: error, 1: warning, 2: info, ...).
        **kwargs: Additional keyword arguments for click.style.
        """
        if self.verbosity < severity:
            return
        if set_indent is not None:
            self.n_indent = set_indent

        if isinstance(message, dict):
            message = self.dict_to_str(message)
        elif isinstance(message, list):
            message = self.list_to_str(message)
        elif isinstance(message, pd.DataFrame | pd.Series):
            message = self.pd_to_str(message)
        else:
            message = self.indent_str * self.n_indent + prepend + message

        message = click.style(message, **kwargs)

        click.echo(message, file=self.file)

        self.n_indent = self.n_indent + indent

    def debug(self, message, indent=0, set_indent=None, severity=3, **kwargs):
        """Print a debug message."""
        self.print(message, indent, set_indent, severity=severity, **kwargs)

    def info(self, message, indent=0, set_indent=None, severity=2, **kwargs):
        """Print a info message."""
        self.print(message, indent, set_indent, severity=severity, **kwargs)

    def start(self, message, indent=0, set_indent=0, severity=2, **kwargs):
        """Print a message in bold at indent 0 to indicate the start of a script"""
        self.print(
            message,
            indent,
            set_indent,
            prepend="🐳 ",
            severity=severity,
            bold=True,
            **kwargs,
        )
        if self.verbosity >= severity:
            self.print(
                f"orcAI {version('orcAI')} [started @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
                indent,
                set_indent,
                severity=severity,
                italic=True,
                **kwargs,
            )

    def part(self, message, indent=1, set_indent=0, severity=2, **kwargs):
        """Print a message in bold at indent 0 to indicate a new part"""

        last_part_time = self.part_times.pop() if len(self.part_times) > 0 else None
        self.part_times.append(time.time())
        total_time = timedelta(seconds=round(self.part_times[-1] - self.start_time))
        delta_time = (
            ", 𝚫 " + str(timedelta(seconds=round(self.part_times[-1] - last_part_time)))
            if last_part_time
            else ""
        )
        if self.show_part_times:
            message = f"{message} [{total_time}{delta_time}]"
        self.print(
            message,
            indent,
            set_indent,
            prepend="🐳 ",
            severity=severity,
            bold=True,
            **kwargs,
        )

    def success(self, message, indent=0, set_indent=0, severity=2, **kwargs):
        """Print a success message."""
        self.part(
            message,
            indent,
            set_indent,
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

    def print_platform_info(self, severity=2, **kwargs):
        if self.verbosity < severity:
            return
        import platform
        import sys

        import keras
        import tensorflow as tf

        self.info(
            f"Platform: {platform.platform()}", severity=severity, italic=True, **kwargs
        )
        self.info(
            f"Python version: {sys.version}", severity=severity, italic=True, **kwargs
        )
        self.info(
            f"Tensorflow version: {tf.__version__}",
            severity=severity,
            italic=True,
            **kwargs,
        )
        self.info(
            f"Keras version: {keras.__version__}",
            severity=severity,
            italic=True,
            **kwargs,
        )
        sys_details = tf.sysconfig.get_build_info()
        if sys_details["is_cuda_build"]:
            self.info(
                f"CUDA version: {sys_details['cuda_version']}",
                severity=severity,
                **kwargs,
            )
            self.info(
                f"cuDNN version: {sys_details['cudnn_version']}",
                severity=severity,
                **kwargs,
            )

    def print_tf_device_info(self, indent=0, set_indent=None, severity=2, **kwargs):
        """print tensorflow devices"""
        if self.verbosity < severity:
            return
        import tensorflow as tf

        tf.get_logger().setLevel(
            40
        )  # suppress tensorflow logging (ERROR and worse only)

        physical_devices = tf.config.list_physical_devices("GPU")
        devices_info = [
            tf.config.experimental.get_device_details(i) for i in physical_devices
        ]

        devices_string = ", ".join(
            [
                f"{dev.name.replace('physical_device:', '')}: {info['device_name']}"
                for dev, info in zip(physical_devices, devices_info)
            ]
        )

        self.info(
            f"Available TensorFlow devices: {devices_string}",
            indent=indent,
            set_indent=set_indent,
            severity=severity,
            italic=True,
            **kwargs,
        )

    def print_memory_usage(self, indent=0, set_indent=None, severity=2, **kwargs):
        """print memory usage"""
        if self.verbosity < severity:
            return
        from psutil import Process

        process = Process(os.getpid())
        self.info(
            f"memory usage: {naturalsize(process.memory_info().rss, format='%.2f')}",
            indent=indent,
            set_indent=set_indent,
            severity=severity,
            italic=True,
            **kwargs,
        )

    def print_file_size(
        self, file: Path, indent=0, set_indent=None, severity=2, **kwargs
    ):
        """print size of dataset"""
        if self.verbosity < severity:
            return

        file_size = Path(file).stat().st_size

        self.info(
            f"Size on disk of {Path(file).name}: {naturalsize(file_size, format='%.2f')}",
            indent=indent,
            set_indent=set_indent,
            severity=severity,
            **kwargs,
        )

    def print_directory_size(
        self, directory: Path, indent=0, set_indent=None, severity=2, **kwargs
    ):
        """print size of directory"""

        if self.verbosity < severity:
            return

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

    def list_to_str(self, list: list) -> str:
        """Convert a list to a formatted string with indentation."""
        list_string = "\n".join(self.indent_str * self.n_indent + line for line in list)
        return list_string

    def dict_to_str(self, dictionary: dict) -> str:
        """Convert a dictionary or list to a formatted string with indentation."""
        json_string = json.dumps(dictionary, indent=4, cls=JsonEncoderExt)
        indented_json = "\n".join(
            self.indent_str * self.n_indent + line for line in json_string.splitlines()
        )
        return indented_json

    def pd_to_str(self, obj: pd.DataFrame) -> str:
        """Convert a DataFrame to a formatted string with indentation."""
        obj_string = obj.to_string()
        indented_obj = "\n".join(
            self.indent_str * self.n_indent + line for line in obj_string.splitlines()
        )
        return indented_obj


def resolve_recording_data_dir(
    recording: str, recording_data_dir: Path | str
) -> Path | None:
    """Resolve the path to the recording data directory.
    Parameter
    ---------
    recording: str
        The name of the recording.
    recording_data_dir: Path | str
        The path to the recording data directory.
    Returns
    -------
        Path | None: The resolved path to the recording data directory, or None if it does not exist.
    """

    if Path(recording_data_dir, recording).exists():
        return Path(recording_data_dir, recording)
    else:
        return None


def filter_filepaths(
    filepaths: list[Path],
    exclude_pattern: list[str],
    msgr: Messenger = Messenger(verbosity=2),
) -> list[Path]:
    """Remove file paths that contain any of the specified patterns.

    Parameter
    ---------
    filepaths: list[Path]
        A list of Path objects representing the file paths to be filtered.
    exclude_pattern: list[str]
        A list of strings, where each string is a pattern to be excluded from the file paths.
    msgr: Messenger
        An instance of the Messenger class used for logging messages with different verbosity levels. Defaults to Messenger(verbosity=2).

    Returns
    -------
        list[Path]: A list of Path objects representing the filtered file paths.

    Example
    -------
        from pathlib import Path
        filepaths = [Path("/path/to/file1.txt"), Path("/path/to/file2.log"), Path("/path/to/file3.txt")]
        exclude_pattern = [".log"]
        filtered_filepaths = filter_filepaths(filepaths, exclude_pattern)
        # filtered_filepaths will be [Path("/path/to/file1.txt"), Path("/path/to/file3.txt")]
    """
    for e in exclude_pattern:
        filepaths = [f for f in filepaths if e not in str(f)]
        msgr.info(
            f"Remaining files after filtering files that contain {e}: {len(filepaths)}"
        )
    return filepaths


def seconds_to_hms(seconds: int) -> str:
    """Convert seconds to hh:mm:ss format

    Parameter
    ---------
    seconds: int
        The number of seconds to convert.
    Returns
    -------
        str: The time in hh:mm:ss format
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def find_consecutive_ones(binary_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the start and end indices of consecutive sequences of ones in a binary vector.

    Parameter
    ---------
    binary_vector: np.ndarray
        A binary vector (1D array of 0s and 1s).
    Returns
    -------
        tuple[np.ndarray, np.ndarray]: (start, end) indices for each sequence of ones.

    """
    # Find where the binary vector changes
    diff = np.diff(binary_vector, prepend=0, append=0)

    # Start indices are where 0 → 1, end indices are where 1 → 0
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0] - 1  # Adjust to include the last 1

    return starts, stops
