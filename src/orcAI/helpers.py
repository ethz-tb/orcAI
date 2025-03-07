import sys
from pathlib import Path
from importlib.resources import files
import pandas as pd

from orcAI.auxiliary import Messenger, filter_filepaths, read_json


def create_recording_table(
    base_dir_recording: Path | str,
    output_path: Path | str | None = None,
    base_dir_annotation: Path | str | None = None,
    default_channel: int = 1,
    label_calls: Path | str | None = None,
    update_table: Path | str | None = None,
    update_paths: bool = True,
    exclude_patterns: Path | str | list[str] | None = None,
    remove_duplicate_filenames: bool = False,
    verbosity: int = 2,
) -> pd.DataFrame:
    """Create a table of recordings for use with other orcAI functions.

    Parameters
    ----------
    base_dir_recording : Path | str
        Base directory containing the recordings (possibly in subdirectories).
    output_path : (Path | str) | None
        Path to save the table of recordings. If none it is saved as recording_table.csv in base_dir_recording.
    base_dir_annotation : (Path | str) | None
        Base directory containing the annotations (if different from base_dir_recording).
    default_channel : int
        Default channel number for the recordings.
    update_table : (Path | str) | None
        Path to a .csv file with a previous table of recordings to update.
    update_paths : bool
        If True the paths in the table to update are updated with the new paths. Only valid if update_table is not None.
    exclude_patterns : (Path | str) | array | None
        Path to a JSON file containing filenames to exclude from the table or an array containing the same.

    Returns
    -------
    recordings_table : pd.DataFrame
        Table of recordings with columns: "channel", "duplicate", "base_dir_recording", "rel_recording_path",
        "base_dir_annotation", "rel_annotation_path", and columns for each call in label_calls. If updating a table
        (ie. update_table is not None), additional columns from the previous table are also included.

    """
    msgr = Messenger(verbosity=verbosity)
    if output_path is None:
        output_path = Path(base_dir_recording).joinpath("recording_table.csv")
    else:
        output_path = Path(output_path)
    if output_path.exists():
        msgr.error(f"Output path {output_path} already exists!")
        sys.exit()
    msgr.part("Creating list of wav files for prediction")

    wav_files = list(Path(base_dir_recording).glob("**/*.wav"))
    if base_dir_annotation is None:
        base_dir_annotation = base_dir_recording
    annotation_files = list(Path(base_dir_annotation).glob("**/*.txt"))

    if exclude_patterns is not None:
        if isinstance(exclude_patterns, (Path | str)):
            exclude_patterns = read_json(exclude_patterns)
        wav_files = filter_filepaths(wav_files, exclude_patterns, msgr=msgr)
        annotation_files = filter_filepaths(
            annotation_files, exclude_patterns, msgr=msgr
        )

    recordings_table = pd.DataFrame(
        data={
            "recording": [path.stem for path in wav_files],
            "recording_type": "unknown",
            "channel": default_channel,
            "base_dir_recording": base_dir_recording,
            "rel_recording_path": [
                path.relative_to(base_dir_recording) for path in wav_files
            ],
        },
    ).set_index("recording")

    annotations_table = pd.DataFrame(
        pd.DataFrame(
            {
                "recording": [path.stem for path in annotation_files],
                "base_dir_annotation": base_dir_annotation,
                "rel_annotation_path": [
                    path.relative_to(base_dir_annotation) for path in annotation_files
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
        ]
    ]

    recordings_table.to_csv(output_path)

    msgr.success("Recordings table created.")
    msgr.info(f"Total number of recordings: {len(recordings_table)}")
    msgr.info(
        f"Total number of unique recordings: {len(recordings_table.index.unique())}"
    )

    return recordings_table
