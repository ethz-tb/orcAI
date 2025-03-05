from pathlib import Path
from importlib.resources import files
import pandas as pd

from orcAI.auxiliary import Messenger, filter_filepaths, read_json, resolve_file_paths


def create_recordings_table(
    base_dir_recordings,
    base_dir_annotations=None,
    default_channel=1,
    label_calls=files("orcAI.defaults").joinpath("default_calls.json"),
    update_table=None,
    update_paths=True,
    exclude_patterns=None,
    remove_duplicate_filenames=False,
    verbosity=2,
):
    """
    Create a table of recordings for use with other orcAI functions.

    Parameters
    ----------
    base_dir_recordings : Path | str
        Base directory containing the recordings (possibly in subdirectories).
    base_dir_annotations : (Path | str) | None
        Base directory containing the annotations (if different from base_dir_recordings).
    default_channel : int
        Default channel number for the recordings.
    label_calls : (Path | str) | dict
        Path to a JSON file containing calls for labeling or a dictionary with calls for labeling.
    update_table : (Path | str) | None
        Path to a .csv file with a previous table of recordings to update.
    update_paths : bool
        If True the paths in the table to update are updated with the new paths. Only valid if update_table is not None.
    exclude_patterns : (Path | str) | array | None
        Path to a JSON file containing filenames to exclude from the table or an array containing the same.

    Returns

    """
    msgr = Messenger(verbosity=verbosity)
    msgr.part("Creating list of wav files for prediction")

    wav_files = list(Path(base_dir_recordings).glob("**/*.wav"))
    if base_dir_annotations is None:
        base_dir_annotations = base_dir_recordings
    annotation_files = list(Path(base_dir_annotations).glob("**/*.txt"))

    if exclude_pattern is not None:
        if isinstance(exclude_pattern, (Path | str)):
            exclude_pattern = read_json(exclude_pattern)
        wav_files = filter_filepaths(wav_files, exclude_pattern, msgr=msgr)
        annotation_files = filter_filepaths(
            annotation_files, exclude_pattern, msgr=msgr
        )

    if isinstance(label_calls, (Path | str)):
        label_calls = read_json(label_calls)

    recordings_table = pd.DataFrame(
        data={
            "recording": [path.stem for path in wav_files],
            "recording_type": "unknown",
            "channel": default_channel,
            "base_dir_recording": base_dir_recordings,
            "rel_recording_path": [
                path.relative_to(base_dir_recordings) for path in wav_files
            ],
            **dict.fromkeys(label_calls, pd.NA),
        },
    ).set_index("recording")

    annotations_table = pd.DataFrame(
        pd.DataFrame(
            {
                "recording": [path.stem for path in annotation_files],
                "base_dir_annotation": base_dir_annotations,
                "rel_annotation_path": [
                    path.relative_to(base_dir_annotations) for path in annotation_files
                ],
            }
        )
    ).set_index("recording")

    recordings_table = recordings_table.join(annotations_table, how="left")
    recordings_table["duplicate"] = recordings_table.index.duplicated(keep=False)

    if recordings_table["duplicate"].any():
        if remove_duplicate_filenames:
            recordings_table = recordings_table[~recordings_table["duplicate"]]
        else:
            msgr.warning("Duplicate filenames found.")
            msgr.warning(
                "Please check the duplicates marked in the output table and ensure file stems"
                + "(filename without extensions) are unique within the specified directories."
            )

    additional_columns = []
    if update_table is not None:
        previous_recordings = pd.read_csv(update_table, index_col="recording")
        additional_columns = previous_recordings.columns.difference(
            recordings_table.columns
        )
        if not update_paths:
            recordings_table[
                [
                    "base_dir_recording",
                    "rel_recording_path",
                    "base_dir_annotation",
                    "rel_annotation_path",
                ]
            ] = None
        recordings_table = recordings_table.combine_first(previous_recordings)

    recordings_table = recordings_table[
        [
            "channel",
            "duplicate",
            "base_dir_recording",
            "rel_recording_path",
            "base_dir_annotation",
            "rel_annotation_path",
            *additional_columns,
            *label_calls,
        ]
    ]

    recordings_table.to_csv(
        "/Users/daniel/polybox/work/projects/orcai_project/test/test_recordings_table.csv"
    )

    msgr.success("Recordings table created.")
    msgr.info(f"Total number of recordings: {len(recordings_table)}")
    msgr.info(
        f"Total number of unique recordings: {len(recordings_table.index.unique())}"
    )

    return recordings_table
