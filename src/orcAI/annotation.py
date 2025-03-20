from pathlib import Path
from importlib.resources import files
import numpy as np
import pandas as pd
from tqdm import tqdm


# import local
from orcAI.auxiliary import (
    Messenger,
    read_json,
    generate_times_from_spectrogram,
    save_as_zarr,
    write_json,
)


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


def convert_annotation(
    annotation_file_path: Path | str,
    recording_data_dir: Path | str,
    labels_present: list,
    labels_masked: list,
    call_equivalences: (Path | str) | dict = None,
    msgr: Messenger = Messenger(),
):
    """Transform annotation into array with 0 for absence and 1 for presence and
    -1 for masked (presence not possible) of each label at times corresponding to spectrogram

    Parameters

    annotation_file_path : Path | str
        Path to the annotation file.
    recording_data_dir : Path | str
        Path to the recording data directory where the spectrogram is stored.
    labels_present : list
        List of labels that are present in the annotation file.
    labels_masked : list
        List of labels that are masked in the annotation file.
    call_equivalences : (Path | str) | dict
        Optional path to a call equivalences file or a dictionary. A dictionary associating original call labels with new call labels
    msgr : Messenger
        Messenger object for logging.

    Returns
    -------
    annotations_array : pd.DataFrame
        DataFrame with columns for each label present and masked in the annotation file.
    label_list : dict
        Dictionary with labels present or masked.

    """
    msgr.part("Converting annotation to label array")
    # read annotation file
    recording = Path(annotation_file_path).stem
    annotations = read_annotation_file(annotation_file_path)

    if call_equivalences is not None:
        msgr.info("Applying call equivalences")
        if isinstance(call_equivalences, (Path | str)):
            call_equivalences = read_json(call_equivalences)
        annotations["label"] = annotations["origlabel"].map(call_equivalences)
        all_orig_labels = set(annotations["origlabel"].unique())
        call_equivalences_keys = set(call_equivalences.keys())
        labels_not_in_equivalences = all_orig_labels.difference(call_equivalences_keys)
        if len(labels_not_in_equivalences) > 0:
            msgr.info("labels not in call_dict:", labels_not_in_equivalences)

    annotations = annotations[["start", "stop", "label"]]
    spectrogram_dir = Path(recording_data_dir).joinpath(recording, "spectrogram")

    # load t_vec of spectrogram
    try:
        t_vec = generate_times_from_spectrogram(spectrogram_dir.joinpath("times.json"))
    except FileNotFoundError as e:
        msgr.error(f"File not found: {spectrogram_dir.joinpath('times.json')}")
        msgr.error("Did you create the spectrogram?")
        raise e

    # Initialize df with label_arr
    annotations_array = pd.DataFrame({})
    # Create a column for each label present
    for label in labels_present:
        # Find all intervals for the current label
        label_intervals = annotations[annotations["label"] == label]

        # Create a boolean mask for the current label
        bool_mask = np.zeros(len(t_vec), dtype=bool)

        # Check if each time step in t_vec is within any interval
        for start, stop in zip(label_intervals["start"], label_intervals["stop"]):
            bool_mask |= (t_vec >= start) & (t_vec <= stop)

        # Add the mask to the result DataFrame as a binary column
        annotations_array[label] = bool_mask.astype(int)

    # Create a column for each label masked
    mask_value = -1
    for label in labels_masked:
        annotations_array[label] = mask_value * np.ones(
            len(t_vec), dtype=int
        )  # set mask value to -1 for label to be masked, set to zero if labels should be assumed absent

    # sort columns alphabetically
    annotations_array = annotations_array.reindex(
        sorted(annotations_array.columns), axis=1
    )

    label_list = dict.fromkeys(labels_present, "present") | dict.fromkeys(
        labels_masked, "masked"
    )
    label_list = dict(sorted(label_list.items()))
    return annotations_array, label_list


def create_label_arrays(
    recording_table_path: Path | str,
    output_dir: Path | str,
    base_dir_annotation: Path | str = None,
    orcai_parameter: (Path | str) | dict = files("orcAI.defaults").joinpath(
        "default_orcai_parameter.json"
    ),
    call_equivalences: (Path | str) | dict = None,
    overwrite: bool = False,
    verbosity: int = 2,
):
    """Makes label arrays for all files in recording_table

    Parameters
    ----------
    recording_table_path : Path | str
        Path to .csv table with columns 'recording', 'channel' and columns corresponding to calls intendend for
        teaching (corresponding to calls in label_call) indicating possibility of presence of calls
        (even if no instance of this call is annotated).
    output_dir : Path | str
        Output directory for the labels. If None the labels are saved in the same directory as the wav files.
    base_dir_annotation : Path
        Base directory for the annotation files. If None the base_dir_annotation is taken from the recording_table.
    orcai_parameter : (Path | str) | dict
        Path to a JSON file containing orcAI parameters or a dictionary of the same.
    call_equivalences : (Path | str) | dict
        Optional path to a call equivalences file or a dictionary. A dictionary associating original call labels with new call labels
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    """
    msgr = Messenger(verbosity=verbosity)
    msgr.part("Making label arrays")

    recording_table = pd.read_csv(recording_table_path)

    if base_dir_annotation is not None:
        recording_table["base_dir_annotation"] = base_dir_annotation

    not_annotated = recording_table["base_dir_annotation"].isna()
    if any(not_annotated):
        msgr.info(f"Skipping {sum(not_annotated)} because of missing annotation files.")
        recording_table = recording_table[~not_annotated]

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)
    recordings_no_labels = []
    label_calls = orcai_parameter["calls"]

    if not overwrite:
        existing_labels = recording_table["recording"].apply(
            lambda x: Path(output_dir).joinpath(x, "labels").exists()
        )
        msgr.info(
            f"Skipping {sum(existing_labels)} recordings because they already have Labels."
        )
        recording_table = recording_table[~existing_labels]

    for i in tqdm(
        recording_table.index,
        desc="Converting annotation files",
        total=len(recording_table),
        unit="recording",
    ):
        recording_labels = recording_table.loc[i, label_calls]
        labels_present = list(recording_labels[recording_labels].index)

        if len(labels_present) > 0:
            labels_masked = list(set(label_calls).difference(labels_present))
            annotations_array, label_list = convert_annotation(
                Path(recording_table.loc[i, "base_dir_annotation"]).joinpath(
                    recording_table.loc[i, "rel_annotation_path"]
                ),
                Path(output_dir),
                labels_present,
                labels_masked,
                call_equivalences=call_equivalences,
                msgr=Messenger(verbosity=0),
            )

            # save
            recording_output_dir = Path(output_dir).joinpath(
                recording_table.loc[i, "recording"], "labels"
            )

            save_as_zarr(
                annotations_array.to_numpy(),
                recording_output_dir.joinpath("labels.zarr"),
                msgr=Messenger(verbosity=0),
            )
            write_json(label_list, recording_output_dir.joinpath("label_list.json"))

        else:
            recordings_no_labels.append(recording_table.loc[i, "recording"])

    if len(recordings_no_labels) > 0:
        msgr.warning(f"No valid labels present in {recordings_no_labels}")
    msgr.success("Finished making label arrays")
    return
