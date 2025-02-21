from pathlib import Path
import numpy as np
import pandas as pd
from click import progressbar
from importlib.resources import files

# import local
import orcAI.auxiliary as aux


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


def read_annotation_files(fns):
    """read multiple annotation file and return with fnstem as additional column"""
    annot = pd.DataFrame()
    for fn in fns:
        a = read_annotation_file(fn)
        annot = pd.concat([annot, a])
    return annot


def apply_label_equivalences(df, dict):
    """map call_dict to origlabel"""
    df["label"] = df["origlabel"].map(dict)
    return df


def convert_annotation(
    annotation_file_path,
    labels_present,
    labels_masked,
    call_equivalences_path=None,
    msgr=aux.Messenger(),
):
    """transform annotation into array with 0 for absence and 1 for presence and -1 for masked (presence not possible) of each label at times t_vec"""
    msgr.part("Converting annotation to label array")
    # read annotation file
    recording = Path(annotation_file_path).stem
    annotations = read_annotation_file(annotation_file_path)

    if call_equivalences_path is not None:
        msgr.info("Applying call equivalences")
        call_equivalences = aux.read_json(call_equivalences_path)
        annotations = apply_label_equivalences(annotations, call_equivalences)
        all_orig_labels = set(annotations["origlabel"].unique())
        call_equivalences_keys = set(call_equivalences.keys())
        labels_not_in_equivalences = all_orig_labels.difference(call_equivalences_keys)
        if len(labels_not_in_equivalences) > 0:
            msgr.info("labels not in call_dict:", labels_not_in_equivalences)

    annotations = annotations[["start", "stop", "label"]]
    spectrogram_dir = Path(annotation_file_path).parent.joinpath(
        recording, "spectrogram"
    )

    # load t_vec of spectrogram
    try:
        t_vec = aux.read_json_to_vector(spectrogram_dir.joinpath("times.json"))
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


def make_label_arrays(
    recording_table_path,
    base_dir=None,
    output_dir=None,
    label_calls_path=files("orcAI.defaults").joinpath("default_calls.json"),
    call_equivalences_path=None,
    verbosity=2,
):
    """Makes label arrays for all files in recording_table_path

    Parameters
    ----------
    recording_table_path : Path
        Path to .csv table with columns 'recording', 'channel' and columns corresponding to calls intendend for
        teaching (corresponding to calls in JSON file at label_calls_path) indicating possibility of presence of calls
        (even if no instance of this call is annotated).
    base_dir : Path
        Base directory for the recording files. If not None entries in the recording column are interpreted as filenames
        searched for in base_dir and subfolders. If None the entries are interpreted as absolute paths.
    output_dir : Path
        Output directory for the labels. If None the labels are saved in the same directory as the wav files.
    label_calls_path : Path
        Path to a JSON file containing calls for labeling
    call_equivalences_path : Path
        Optional path to a call equivalences file. A dictionary associating original call labels with new call labels
    verbosity : int
        Verbosity level.
    """
    msgr = aux.Messenger(verbosity=verbosity)
    msgr.part("OrcAI - Making label arrays")

    recording_table = pd.read_csv(recording_table_path)
    if base_dir is not None:
        msgr.info(f"Resolving file paths...")
        recording_table["annotation_file"] = aux.resolve_file_paths(
            base_dir,
            recording_table["recording"],
            ".txt",
            msgr=msgr,
        )

    missing_annotations = pd.isna(recording_table["annotation_file"])
    if any(missing_annotations):
        msgr.warning(
            f"Missing annotation files for {sum(missing_annotations)} recordings. Skipping these recordings."
        )
        recording_table = recording_table[~missing_annotations]

    label_calls = aux.read_json(label_calls_path)
    recordings_no_labels = []
    with progressbar(
        recording_table.index,
        label="Converting annotation files...",
        item_show_func=lambda index: aux._recording_table_show_func(
            index,
            recording_table,
        ),
    ) as recording_indices:
        for i in recording_indices:
            recording_labels = recording_table.loc[i, label_calls]
            labels_present = list(recording_labels[recording_labels].index)

            if len(labels_present) > 0:
                labels_masked = list(set(label_calls).difference(labels_present))
                annotations_array, label_list = convert_annotation(
                    recording_table.loc[i, "annotation_file"],
                    labels_present,
                    labels_masked,
                    call_equivalences_path=call_equivalences_path,
                    msgr=aux.Messenger(verbosity=0),
                )

                # save to as numpy
                if output_dir is None:
                    output_dir = recording_table.loc[
                        i, "annotation_file"
                    ].parent.joinpath(recording_table.loc[i, "recording"], "labels")
                else:
                    output_dir = Path(output_dir).joinpath("labels")

                aux.save_as_zarr(
                    annotations_array.to_numpy(), output_dir.joinpath("zarr.lbl")
                )
                aux.write_json(label_list, output_dir.joinpath("label_list.json"))

            else:
                recordings_no_labels.append(recording_table.loc[i, "recording"])

    msgr.warning(f"No labels present in {recordings_no_labels}")
    return


def reshape_label_arr(arr, n_filters):
    dim1 = arr.shape[0] // n_filters
    dim2 = arr.shape[1]
    if arr.shape[0] % n_filters == 0:
        arr_out = (arr.reshape(dim1, n_filters, dim2).mean(axis=1) + 0.5).astype(int)
    return arr_out
