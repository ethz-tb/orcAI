from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from orcAI.auxiliary import MASK_VALUE, Messenger
from orcAI.io import (
    generate_times_from_spectrogram,
    read_annotation_file,
    read_json,
    save_as_zarr,
    write_json,
)


def _convert_annotation(
    annotation_file_path: Path,
    recording_data_dir: Path,
    label_calls: list,
    labels_present: list,
    labels_masked: list,
    call_equivalences: (Path | str) | dict = None,
    msgr: Messenger = Messenger(),
) -> tuple[pd.DataFrame, dict]:
    """Transform annotation into array with 0 for absence and 1 for presence and
    MASK_VALUE for masked (presence not possible) of each label at times corresponding to spectrogram

    Parameter

    annotation_file_path : Path
        Path to the annotation file.
    recording_data_dir : Path
        Path to the recording data directory where the spectrogram is stored.
    label_calls : list
        List of labels that are intended for teaching.
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

    Raises
    ------
    FileNotFoundError
        If the spectrogram file is not found.
    """
    msgr.part("Converting annotation to label array")
    # read annotation file
    recording = annotation_file_path.stem
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
            msgr.info("labels not in call equivalences:", labels_not_in_equivalences)

    annotations = annotations[["start", "stop", "label"]]
    spectrogram_dir = recording_data_dir.joinpath(recording, "spectrogram")

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
    for label in labels_masked:
        annotations_array[label] = MASK_VALUE * np.ones(
            len(t_vec), dtype=int
        )  # set mask value for label to be masked

    # sort columns in original order
    annotations_array = annotations_array.reindex(label_calls, axis=1)

    label_dict = dict.fromkeys(labels_present, "present") | dict.fromkeys(
        labels_masked, "masked"
    )
    # sort label_dict in original order
    label_dict = {k: label_dict[k] for k in label_calls}
    return annotations_array, label_dict


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
    msgr: Messenger | None = None,
) -> None:
    """Makes label arrays for all files in recording_table

    Parameter
    ----------
    recording_table_path : Path | str
        Path to .csv table with columns 'recording', 'channel' and columns corresponding to calls intendend for
        teaching (corresponding to calls in label_call) indicating possibility of presence of calls
        (even if no instance of this call is annotated).
    output_dir : Path | str
        Output directory for the labels. Labels are stored in subdirectories named '<recording>/labels'
    base_dir_annotation : Path
        Base directory for the annotation files. If None the base_dir_annotation is taken from the recording_table.
    orcai_parameter : (Path | str) | dict
        Path to a JSON file containing orcAI parameter or a dictionary of the same.
    call_equivalences : (Path | str) | dict
        Optional path to a call equivalences file or a dictionary. A dictionary associating original call labels with new call labels
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.

    Returns
    -------
    None
        Creates label arrays and saves them in the specified output directory in named '<recording>/labels'.
    """
    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Making label arrays")

    msgr.part("Reading recordings table")
    output_dir = Path(output_dir)

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
            lambda x: output_dir.joinpath(x, "labels").exists()
        )
        msgr.info(
            f"Skipping {sum(existing_labels)} recordings because they already have Labels."
        )
        recording_table = recording_table[~existing_labels]

    msgr.part("Making label arrays")
    for i in tqdm(
        recording_table.index,
        desc="Making label arrays",
        total=len(recording_table),
        unit="recording",
    ):
        recording_labels = recording_table.loc[i, label_calls]
        labels_present = list(recording_labels[recording_labels].index)

        if len(labels_present) > 0:
            labels_masked = list(set(label_calls).difference(labels_present))
            annotations_array, label_dict = _convert_annotation(
                annotation_file_path=Path(
                    recording_table.loc[i, "base_dir_annotation"]
                ).joinpath(recording_table.loc[i, "rel_annotation_path"]),
                recording_data_dir=output_dir,
                label_calls=label_calls,
                labels_present=labels_present,
                labels_masked=labels_masked,
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
            )
            write_json(label_dict, recording_output_dir.joinpath("label_list.json"))

        else:
            recordings_no_labels.append(recording_table.loc[i, "recording"])

    if len(recordings_no_labels) > 0:
        msgr.warning(f"No valid labels present in {recordings_no_labels}")
    msgr.success("Finished making label arrays")
    return
