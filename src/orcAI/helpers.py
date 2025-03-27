import sys
import shutil
from pathlib import Path
from importlib.resources import files
import pandas as pd
from numpy.random import SeedSequence

from orcAI.auxiliary import Messenger, filter_filepaths
from orcAI.io import read_json, write_json


def init_project(
    project_dir: Path | str,
    project_name: str,
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> None:
    """Initialize a new orcAI project.

    Parameters
    ----------
    project_dir : Path | str
        Path to the project directory.
    project_name : str
        Name of the project.
    verbosity : int
        Verbosity level. 0: only errors, 1: only warnings, 2: info, 3: debug.
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.

    Returns
    -------
    None
        Creates new project directory with default configuration files.
    """
    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Initializing project")
    msgr.part(f"Creating project directory: {project_dir}")
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Copy the default configuration files
    default_files = files("orcAI.defaults").iterdir()
    for file in default_files:
        new_file_path = project_dir.joinpath(file.name.replace("default", project_name))
        msgr.info(f"Creating {new_file_path.name}")
        shutil.copy(
            file,
            new_file_path,
        )

    orcai_parameter_new = read_json(
        project_dir.joinpath(
            "default_orcai_parameter.json".replace("default", project_name)
        )
    )
    msgr.info(f"Setting seed")
    orcai_parameter_new["seed"] = SeedSequence().generate_state(1)[0]
    orcai_parameter_new["name"] = project_name
    write_json(
        orcai_parameter_new,
        project_dir.joinpath(
            "default_orcai_parameter.json".replace("default", project_name)
        ),
    )
    msgr.success("Project initialized.")


def create_recording_table(
    base_dir_recording: Path | str,
    output_path: Path | str | None = None,
    base_dir_annotation: Path | str | None = None,
    default_channel: int = 1,
    orcai_parameter: Path | str | None = None,
    update_table: Path | str | None = None,
    update_paths: bool = True,
    exclude_patterns: Path | str | list[str] | None = None,
    remove_duplicate_filenames: bool = False,
    verbosity: int = 2,
    msgr: Messenger | None = None,
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
    orcai_parameter : (Path | str) | None
        Path to a JSON file containing orcAi parameter.
    update_table : (Path | str) | None
        Path to a .csv file with a previous table of recordings to update.
    update_paths : bool
        If True the paths in the table to update are updated with the new paths. Only valid if update_table is not None.
    exclude_patterns : (Path | str) | array | None
        Path to a JSON file containing filenames to exclude from the table or an array containing the same.
    remove_duplicate_filenames : bool
        If True, remove duplicate filenames from the table.
    verbosity : int
        Verbosity level. 0: only errors, 1: only warnings, 2: info, 3: debug.
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.

    Returns
    -------
    recordings_table : pd.DataFrame
        Table of recordings with columns: "channel", "duplicate", "base_dir_recording", "rel_recording_path",
        "base_dir_annotation", "rel_annotation_path". If label_calls is given, a column for each call is included.
        If updating a table (ie. update_table is not None), additional columns from the previous table are also included.

    """
    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Creating recording table")

    msgr.part("Resolving file paths")
    if output_path is None:
        output_path = Path(base_dir_recording).joinpath("recording_table.csv")
    else:
        output_path = Path(output_path)
    if output_path.exists():
        msgr.error(f"Output path {output_path} already exists!")
        sys.exit()

    wav_files = list(Path(base_dir_recording).glob("**/*.wav"))

    if base_dir_annotation is None:
        base_dir_annotation = base_dir_recording
    annotation_files = list(Path(base_dir_annotation).glob("**/*.txt"))

    if exclude_patterns is not None:
        if isinstance(exclude_patterns, (Path | str)):
            exclude_patterns = read_json(exclude_patterns)
        msgr.part(f"Filtering {len(wav_files)} wav files...")
        wav_files = filter_filepaths(wav_files, exclude_patterns, msgr=msgr)
        msgr.part(
            f"Filtering {len(annotation_files)} annotations files...",
        )
        annotation_files = filter_filepaths(
            annotation_files, exclude_patterns, msgr=msgr
        )

    if orcai_parameter is not None:
        label_calls = read_json(orcai_parameter)["calls"]
        call_possible = {call: pd.NA for call in label_calls}
    else:
        call_possible = {}

    recording_table = pd.DataFrame(
        data={
            "recording": [path.stem for path in wav_files],
            "recording_type": "unknown",
            "channel": default_channel,
            "base_dir_recording": base_dir_recording,
            "rel_recording_path": [
                path.relative_to(base_dir_recording) for path in wav_files
            ],
            **call_possible,
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

    missing_recording = set(annotations_table.index) - set(recording_table.index)
    if len(missing_recording) > 0:
        msgr.warning(
            f"{len(missing_recording)} annotations with missing recordings: {missing_recording}. These will be ignored."
        )

    recording_table = recording_table.join(annotations_table, how="left")
    recording_table["duplicate"] = recording_table.index.duplicated(keep=False)
    n_duplicates = recording_table["duplicate"].sum()
    if n_duplicates > 0:
        if remove_duplicate_filenames:
            recording_table = recording_table[~recording_table["duplicate"]]
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
            recording_table.columns
        )
        if not update_paths:
            recording_table[
                [
                    "base_dir_recording",
                    "rel_recording_path",
                    "base_dir_annotation",
                    "rel_annotation_path",
                ]
            ] = None
        recording_table = recording_table.combine_first(previous_recordings)

    recording_table = recording_table[
        [
            "channel",
            "duplicate",
            "base_dir_recording",
            "rel_recording_path",
            "base_dir_annotation",
            "rel_annotation_path",
            *additional_columns,
            *call_possible.keys(),
        ]
    ]

    msgr.part(f"Saving recording table to {output_path.relative_to(Path.cwd())}")
    recording_table.to_csv(output_path)
    msgr.info(
        f"Total recordings: {len(recording_table)}",
        set_indent=1,
    )
    if n_duplicates > 0:
        msgr.info(
            f"Number of duplicate recordings: {recording_table['duplicate'].sum()}"
        )
    msgr.info(
        f"Total recordins with annotations: {recording_table['rel_annotation_path'].count()}"
    )

    msgr.success("Recordings table created.")

    return recording_table
