import pandas as pd
import numpy as np
import librosa
import os
from pathlib import Path
import json
import glob
import librosa
from sklearn.metrics import confusion_matrix
import click


def write_vector_to_json(vector, filename):
    """write out equally spaced vector in short form with min, max and length"""
    dictionary = {"min": vector[0], "max": vector[-1], "length": len(vector)}
    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=4)
    return


def read_json_to_vector(filename):
    """read and generate equally spaced vector in short form from min, max and length and"""
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


def filter_filenames(files, eliminate):
    """remove filenames containing patterns in list eliminate"""
    print("# of all files:", len(files))
    for e in eliminate:
        files = [f for f in files if not e in f]
        print("# after taking out files that contain", e, ":", len(files))
    return files


def wav_and_annot_files(computer, directories, eliminate):
    """
    Get names of all annot and wav files on computer and eliminate files based on eliminate
    Args:
        computer (str): name of computer on which program is executed
        directories (dict): directory paths used on that computer
        eliminate (list of str): list on the basis of which files are eliminated
    Returns:
        all_annot_files: list of all valid annotation files on this computer
        fnstem_annotfile_dict: dict that links fnstem to filename
        all_wav_files: list of all valid wav files on this computer
        fnstem_wavfile_dict: dict that links fnstem with filemane
    """

    root_dir_acoustics = directories[computer]["root_dir_acoustics"]

    all_txt_files = get_all_files_with_ext(root_dir_acoustics, ".txt")
    all_wav_files = get_all_files_with_ext(root_dir_acoustics, ".wav")

    filtered_txt_files = filter_filenames(all_txt_files, eliminate)
    filtered_wav_files = filter_filenames(all_wav_files, eliminate)

    fnstem_txt = [Path(f).stem for f in filtered_txt_files]
    fnstem_wav = [Path(f).stem for f in filtered_wav_files]

    print("# WARNING: fnstems for which we have .txt but no .wav file")
    fnstems_to_exclude_from_annot_files = list(
        set(fnstem_txt).difference(set(fnstem_wav))
    )
    print(fnstems_to_exclude_from_annot_files)
    print("  - removing these files from list of .txt")
    filtered_txt_files = [
        f
        for f in filtered_txt_files
        if not any(fnstem in f for fnstem in fnstems_to_exclude_from_annot_files)
    ]

    fnstem_wav_mono = [f.replace("mono", "") for f in fnstem_wav if "mono" in f]
    fnstem_wav_not_mono = [f for f in fnstem_wav if "mono" not in f]
    filtered_wav_files = [
        f for f in filtered_wav_files if "mono" not in f
    ]  # eliminate wav files containing mono

    fnstems_not_without_mono = list(
        set(fnstem_wav_mono).difference(set(fnstem_wav_not_mono))
    )
    print(
        '# fnstems containing "mono" but not found as fnstems without "mono":',
        fnstems_not_without_mono,
    )

    fnstems_wav_but_not_txt = set(fnstem_wav_not_mono).difference(set(fnstem_txt))
    print(
        '# fnstems for which we have wav but no .txt file excluding files containing "mono":',
        len(fnstems_wav_but_not_txt),
    )

    # remove files where wav cannot be read
    filtered_wav_files_wo_ext = [f.replace(".wav", "") for f in filtered_wav_files]
    filtered_txt_files_wo_ext = [f.replace(".txt", "") for f in filtered_txt_files]

    files_corrupt = []
    for fstem in list(set(filtered_wav_files_wo_ext + filtered_txt_files_wo_ext)):
        try:
            librosa.get_duration(path=fstem + ".wav")
        except:
            print("WARNING: problems reading file\n", fstem + ".wav")
            files_corrupt = files_corrupt + [fstem]
    print("\n - eliminating", len(files_corrupt), "corrupt wav files from list")
    filtered_wav_files_wo_ext = list(
        set(filtered_wav_files_wo_ext) - set(files_corrupt)
    )
    filtered_wav_files = [f + (".wav") for f in filtered_wav_files_wo_ext]
    filtered_txt_files_wo_ext = list(
        set(filtered_txt_files_wo_ext) - set(files_corrupt)
    )
    filtered_txt_files = [f + (".txt") for f in filtered_txt_files_wo_ext]

    print(
        "# annotation files with non corrupt wav files associated:",
        len(filtered_txt_files),
        "\n# non corrupt wav files:",
        len(filtered_wav_files),
    )

    all_wav_files = filtered_wav_files
    fnstem_wavfile_dict = {}
    for fn in filtered_wav_files:
        fnstem = Path(fn).stem
        fnstem_wavfile_dict[fnstem] = fn

    all_annot_files = filtered_txt_files
    fnstem_annotfile_dict = {}
    for fn in filtered_txt_files:
        fnstem = Path(fn).stem
        fnstem_annotfile_dict[fnstem] = fn
    return all_annot_files, fnstem_annotfile_dict, all_wav_files, fnstem_wavfile_dict


# get wav duration
def get_wav_duration(files):
    """get duration of all wav files in files"""
    fnstems = []
    duration = []
    for f in files:
        try:
            dur = librosa.get_duration(path=f)
            fnstems += [Path(f).stem]
            duration += [dur]
        except:
            print("WARNING: wav file cannot be read:", f)
    return pd.DataFrame({"fnstem": fnstems, "duration": duration})


# total duration for each label in each fnstem
def calculate_total_duration(annotations):
    """
    Calculate the total duration for each label within each fnstem,
    and return a wide-format DataFrame.

    Args:
        annotations (pd.DataFrame): A DataFrame with columns ['fnstem', 'start', 'stop', 'label'].

    Returns:
        pd.DataFrame: A wide-format DataFrame with one row per fnstem and columns for each label.
    """
    # Calculate duration for each row
    annotations["duration"] = annotations["stop"] - annotations["start"]

    # Group by 'fnstem' and 'label' and sum the durations
    grouped = annotations.groupby(["fnstem", "label"])["duration"].sum().reset_index()

    # Pivot to wide format
    wide_format = grouped.pivot(
        index="fnstem", columns="label", values="duration"
    ).fillna(0)

    # Reset index to make 'fnstem' a column
    wide_format.reset_index(inplace=True)

    # Rename columns to include 'label' prefix for clarity
    wide_format.rename(columns=lambda x: f"{x}" if x != "fnstem" else x, inplace=True)

    # Add a row for the column sums
    column_sums = wide_format.iloc[:, 1:].sum(axis=0)  # Sum durations for each label
    total_row = pd.DataFrame(
        [["Total"] + column_sums.tolist()], columns=wide_format.columns
    )
    wide_format = pd.concat([total_row, wide_format], ignore_index=True)

    return wide_format


# TODO: remove
def check_interactive():
    """check if running interactive or not"""

    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is not None:
            print("Interactive session for IPython or Jupyter environment.")
            interactive = True
            # Load the autoreload extension
            ipython.run_line_magic("load_ext", "autoreload")
            # Set autoreload mode to 2, which automatically reloads modules when they change
            ipython.run_line_magic("autoreload", "2")
            print("Autoreload enabled.")
        else:
            interactive = False
            print("Not running in an IPython or Jupyter environment.")
    except ImportError:
        interactive = False
        print("Not running in an IPython or Jupyter environment.")
    return interactive


def get_annot_and_wav_files(root_dir, computer):
    # get annotation and wav file names
    print("GETTING ALL WAV AND ANNOTATION FILE NAMES BELOW", root_dir)
    print(
        "  - NOTE: filenames must be named [filename].txt and [filename].wav and no filename may occur twice across all directories"
    )
    print(
        "  - getting all annotation (.txt) files",
    )
    all_txt_files = list(glob.iglob(root_dir + "**/*.txt", recursive=True))
    print("    -", len(all_txt_files), "annotation files found")
    print("  - getting all audio (.wav) files ")
    all_wav_files = list(glob.iglob(root_dir + "**/*.wav", recursive=True))
    print("    -", len(all_wav_files), "audio files found")

    # keeping only files for which there is a .txt corresponding to a *.wav
    print("  - keeping only files were there is a .txt and .wav with identical name")
    xs = [x.replace(".txt", "") for x in all_txt_files] + [
        x.replace(".wav", "") for x in all_wav_files
    ]
    all_annot_files = list(set([x + ".txt" for x in xs if xs.count(x) > 1]))
    elim = ["_ChB", "_Chb", "Movie", "Norway", "_acceleration", "_depthtemp"]
    print("  - eliminating files that contain", elim, " in directory or filename:")
    all_annot_files = [
        elem
        for elem in all_annot_files
        if not any(substring in elem for substring in elim)
    ]
    print("    -", len(all_annot_files), "files remaining")

    # keeping only files that have a wav file that can be read
    import librosa

    print("  - eliminating files (.txt and.wav) where wav cannot be read")
    files_corrupt = []
    for f in all_annot_files:
        fwav = f.replace(".txt", ".wav")
        try:
            t_tot = librosa.get_duration(path=fwav)
        except:
            print("WARNING: problems reading file\n", fwav)
            files_corrupt = files_corrupt + [f]
    print("\n - eliminating", len(files_corrupt), "corrupt wav files from list")
    all_annot_files = list(set(all_annot_files) - set(files_corrupt))
    print(
        "    -",
        len(all_annot_files),
        "annotation/audio files pairs with readable corresponding wav file",
    )
    fnstem_wavfile_dict = {}
    for fn in all_wav_files:
        fnstem = Path(fn).stem
        fnstem_wavfile_dict[fnstem] = fn
    fnstem_annotfile_dict = {}
    for fn in all_annot_files:
        fnstem = Path(fn).stem
        fnstem_annotfile_dict[fnstem] = fn

    return all_annot_files, fnstem_annotfile_dict, all_wav_files, fnstem_wavfile_dict


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


# TODO: move to paper code
def latex_confusion_matrices(confusion_matrices):
    print("\\begin{table}[h]\n\\centering\n\\begin{tabular}{c c}\n")
    for label, cms in confusion_matrices.items():
        cm1 = cms["cm"]
        cm2 = cms["cm_all"]
        print("\\begin{tabularx}{0.5\\textwidth}{|l|c|c|}")
        print("\\hline")
        print(
            "\\multicolumn{3}{|c|}{\\", label, ", total=", cm1["Total"], "}\\\\", sep=""
        )
        print("\\hline")
        print("  & pred POS & pred NEG \\\\")
        print("\\hline")
        print(
            "act POS  & ",
            np.around(100 * cm1["TP"], 5),
            "&",
            np.around(100 * cm1["FN"], 5),
            "\\\\",
        )
        print("\\hline")
        print(
            "act NEG  & ",
            np.around(100 * cm1["FP"], 5),
            "&",
            np.around(100 * cm1["TN"], 5),
            "\\\\",
        )
        print("\\hline")
        print("\\end{tabularx}")

        print("&")

        print("\\begin{tabularx}{0.5\\textwidth}{|l|c|c|}")
        print("\\hline")
        print(
            "\\multicolumn{3}{|c|}{\\", label, ", total=", cm2["Total"], "}\\\\", sep=""
        )
        print("\\hline")
        print("  & pred POS & pred NEG \\\\")
        print("\\hline")
        print(
            "act POS  & ",
            np.around(100 * cm2["TP"], 5),
            "&",
            np.around(100 * cm2["FN"], 5),
            "\\\\",
        )
        print("\\hline")
        print(
            "act NEG  & ",
            np.around(100 * cm2["FP"], 5),
            "&",
            np.around(100 * cm2["TN"], 5),
            "\\\\",
        )
        print("\\hline")
        print("\\end{tabularx} \\\\")

    print("\\end{tabular}")
    print("\\end{table}\n")

    return


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


class Messenger:
    """Class for printing messages with different levels of verbosity and indentation"""

    def __init__(self, n_indent=0, verbosity=2, indent_str="    ", file=None):
        """
        Initialize the Messenger object with the specified verbosity level and indentation.

        Parameters:
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
        if self.verbosity <= severity:
            return
        if set_indent is not None:
            self.n_indent = set_indent

        if isinstance(message, dict) or isinstance(message, list):
            message = self.dict_to_str(message)
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
        import psutil

        process = psutil.Process(os.getpid())
        self.info(
            f"memory usage: {process.memory_info().rss / 1024 ** 2} MB",
            indent=indent,
            set_indent=set_indent,
            severity=severity,
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
        json_string = json.dumps(dictionary, indent=4)
        indented_json = "\n".join(
            self.indent_str * self.n_indent + line for line in json_string.splitlines()
        )
        return indented_json
