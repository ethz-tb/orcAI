from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import zarr
from tqdm import tqdm

from orcAI.auxiliary import (
    SEED_ID_CREATE_DATALOADER,
    SEED_ID_FILTER_SNIPPET_TABLE,
    SEED_ID_MAKE_SNIPPET_TABLE,
    Messenger,
    resolve_recording_data_dir,
    seconds_to_hms,
)
from orcAI.io import DataLoader, read_json, save_dataset, write_json

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)


def _make_snippet_table(
    recording_dir: Path | str,
    orcai_parameter: dict,
    rng=np.random.default_rng(),
    msgr: Messenger = Messenger(verbosity=2),
) -> tuple[pd.DataFrame, int, int, str, str]:
    """Generates times for snippets to be extracted from recordings

    returns pd.DataFrame with recording, data_type, start, stop and duration for each label

    Parameters
    ----------
    recording_dir : Path | str
        Path to the recording data directory
    orcai_parameter : dict
        dict containing orcai parameter
    msgr : Messenger
        Messenger object for messages

    Returns
    -------
    snippet_table: pd.DataFrame
        snippet table with columns recording, data_type, row_start, row_stop and call names
    recording_duration: int
        recording duration
    n_segments: int
        number of segments
    recording: str
        recording name
    status: str
        status of snippet table creation (success or reason for failure)
    """
    recording = Path(recording_dir).stem
    label_zarr_path = Path(recording_dir).joinpath("labels", "labels.zarr")
    label_list_path = Path(recording_dir).joinpath("labels", "label_list.json")
    spectrogram_times_path = Path(recording_dir).joinpath("spectrogram", "times.json")

    try:
        spectrogram_times = read_json(spectrogram_times_path)
    except FileNotFoundError as _:
        msgr.error(f"File not found: {spectrogram_times_path}")
        msgr.error("Did you create the spectrogram?")
        raise

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)
    model_parameter = orcai_parameter["model"]
    snippet_parameter = orcai_parameter["snippets"]

    recording_duration = spectrogram_times["max"]
    n_segments = int(recording_duration // snippet_parameter["segment_duration"])
    if n_segments <= 0:
        msgr.warning(
            f"Duration of recording ({recording_duration}) is shorter than segment length ({snippet_parameter['segment_duration']}). Skipping recording."
        )
        return (
            None,
            recording_duration,
            n_segments,
            recording,
            "shorter than segment_duration",
        )

    try:
        label_filepointer = zarr.open(label_zarr_path, mode="r")
    except FileNotFoundError as _:
        msgr.warning(f"Label file not found: {label_zarr_path}")
        return (None, recording_duration, n_segments, recording, "missing label files")
    try:
        label_list = read_json(label_list_path)
    except FileNotFoundError as _:
        msgr.warning(f"Label file not found: {label_list_path}")
        return (None, recording_duration, n_segments, recording, "missing label files")

    label_names = list(label_list.keys())

    times = np.linspace(
        spectrogram_times["min"],
        spectrogram_times["max"],
        spectrogram_times["length"],
    )
    delta_t = times[1] - times[0]
    n_filters = len(model_parameter["filters"])
    n_spectrogram_snippet_steps = int(
        (2**n_filters)
        * ((snippet_parameter["snippet_duration"] / delta_t) // (2**n_filters))
    )  # to make time axis divisible by 2 ** n_filters
    msgr.info(f"Number of spectrogram snippet timesteps: {n_spectrogram_snippet_steps}")
    snippet_table_raw = []

    for i_segment in range(n_segments):  # iterate over all segments
        msgr.info(f"Segment {i_segment + 1} of {n_segments}")
        slice = (0, 0)
        for type in list(["train", "val", "test"]):  # iterate over type of snippet
            slice = (slice[1], slice[1] + snippet_parameter[type])
            t_min = (i_segment + slice[0]) * snippet_parameter["segment_duration"]
            for j in range(
                int(
                    snippet_parameter[type]
                    * snippet_parameter["segment_duration"]
                    * snippet_parameter["snippets_per_sec"]
                )
            ):  # iterate over number of snippets per segment and type
                t_max = (i_segment + slice[1]) * snippet_parameter[
                    "segment_duration"
                ] - snippet_parameter["snippet_duration"]
                t_start = rng.uniform(low=t_min, high=t_max, size=1)[0]

                # Find the max index where entries are smaller than t_start
                index_t_start = np.searchsorted(times, t_start, side="left") - 1
                # Find the min index where entries are smaller or equal to t_stop
                index_t_stop = index_t_start + n_spectrogram_snippet_steps
                label_chunk = label_filepointer[index_t_start:index_t_stop, :]
                label_duration_snippet = label_chunk.sum(axis=0) * delta_t
                label_duration_snippet[label_duration_snippet < 0] = np.nan
                snippet_table_raw += [
                    list(
                        [
                            recording,
                            recording_dir,
                            type,
                            index_t_start,
                            index_t_stop,
                        ]
                    )
                    + list(label_duration_snippet)
                ]
    snippet_table = pd.DataFrame(
        snippet_table_raw,
        columns=[
            "recording",
            "recording_data_dir",
            "data_type",
            "row_start",
            "row_stop",
        ]
        + label_names,
    )
    return (snippet_table, recording_duration, n_segments, recording, "success")


def _compute_snippet_stats(
    snippet_table: pd.DataFrame, for_calls: list
) -> pd.DataFrame:
    """Compute snippet stats for calls

    Parameters
    ----------
    snippet_table : pd.DataFrame
        snippet_table with columns recording, data_type, row_start, row_stop and call names
    for_calls : list
        list of call names to compute stats for

    Returns
    -------
    pd.DataFrame
        snippet stats with columns for_calls, total and for_calls_ef
    """

    snippet_stats = snippet_table.groupby("data_type")[for_calls].sum().T
    snippet_stats["total"] = snippet_stats.sum(axis=1)

    equalizing_factors = snippet_stats.apply(lambda x: 1 / x * x.max(), axis=0)
    equalizing_factors.columns = equalizing_factors.columns + "_ef"

    return pd.merge(
        snippet_stats, equalizing_factors, left_index=True, right_index=True
    )


def create_snippet_table(
    recording_table_path: Path | str,
    recording_data_dir: Path | str,
    output_dir: Path | str = None,
    orcai_parameter: dict | (Path | str) = files("orcAI.defaults").joinpath(
        "default_orcai_parameter.json"
    ),
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> None:
    """Generates snippet table for all recordings in recording_table and saves it to disk

    Parameters
    ----------
    recording_table_path : (Path | str)
        Path to the recording table
    recording_data_dir : (Path | str)
        Path to the recording data directory
    output_dir : (Path | str)
        Path to the output directory. If None the output_dir is set to "tvt_data" next to the recording_table_path
    orcai_parameter : dict | (Path | str)
        Dict containing OrcAI parameter or path to json containing the same, by default files("orcAI.defaults").joinpath("default_orcai_parameter.json")
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.

    Returns
    -------
    None. Writes snippet table to disk
    """
    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Making snippet table")

    msgr.part("Reading recording table")

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)

    if output_dir is None:
        output_dir = Path(recording_table_path).parent.joinpath("tvt_data")
    else:
        output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    recording_data_dir = Path(recording_data_dir)
    recording_table_path = Path(recording_table_path)
    recording_table = pd.read_csv(recording_table_path)
    # remove recordings without annotation
    not_annotated = recording_table["base_dir_annotation"].isna()
    recording_table = recording_table[~not_annotated]

    recording_table["recording_data_dir"] = recording_table.apply(
        lambda row: resolve_recording_data_dir(row["recording"], recording_data_dir),
        axis=1,
    )

    missing_recording_data_dir = pd.isna(recording_table["recording_data_dir"])
    if any(missing_recording_data_dir):
        msgr.warning(
            f"Missing recording data directories for {sum(missing_recording_data_dir)} recordings. Skipping these recordings."
        )
        msgr.warning("Did you create the spectrograms & Labels?")
        recording_table = recording_table[~missing_recording_data_dir]

    recording_lengths = []
    segments = []
    all_snippet_tables = []
    failed = []
    failed_result = []
    msgr.part("Making snippet tables")
    rng = np.random.default_rng(
        seed=[SEED_ID_MAKE_SNIPPET_TABLE, orcai_parameter["seed"]]
    )
    for i in tqdm(
        recording_table.index,
        desc="Making snippet tables",
        total=len(recording_table),
        unit="recording",
    ):
        snippet_table, recording_length, n_segments, recording, result = (
            _make_snippet_table(
                recording_table.loc[i, "recording_data_dir"],
                orcai_parameter=orcai_parameter,
                rng=rng,
                msgr=Messenger(verbosity=0),
            )
        )
        if result == "success":
            all_snippet_tables.append(snippet_table)
            recording_lengths.append(recording_length)
            segments.append(n_segments)
        else:
            failed.append(recording)
            failed_result.append(result)

    snippet_table = pd.concat(all_snippet_tables).reset_index(drop=True)
    failed_table = pd.DataFrame({"recording": failed, "reason": failed_result})

    msgr.info(
        f"Created snippet table for {len(snippet_table['recording'].unique())} recordings."
    )
    msgr.info(f"Total recording duration: {seconds_to_hms(np.sum(recording_lengths))}.")
    msgr.info(f"Total number of snippets: {len(snippet_table)}.")
    msgr.info(f"Total number of segments: {np.sum(segments)}")

    msgr.info(f"Creating snippet table failed for {len(failed)} recordings.", indent=1)
    msgr.info(failed_table.groupby("reason").size(), indent=-1)

    msgr.part("Saving snippet table...")

    failed_table.to_csv(
        output_dir.joinpath("failed_snippets.csv"),
        index=False,
    )

    snippet_table.to_csv(
        output_dir.joinpath("all_snippets.csv.gz"),
        compression="gzip",
        index=False,
    )
    msgr.success(f"Snippet table saved to {output_dir.joinpath('all_snippets.csv.gz')}")

    return


def _filter_snippet_table(
    snippet_table: pd.DataFrame,
    orcai_parameter: dict,
    rng=np.random.default_rng(),
    msgr: Messenger = Messenger(verbosity=2),
) -> pd.DataFrame:
    """Filters snippet table based on snippet parameter and label calls

    Parameters
    ----------
    snippet_table : pd.DataFrame
        snippet_table with columns recording, data_type, row_start, row_stop and call names
    orcai_parameter : dict
        dict containing orcai parameter
    msgr : Messenger
        Messenger object for messages

    Returns
    -------
    snippet_table: pd.DataFrame
        filtered snippet table
    """
    msgr.part("Filtering snippet table")

    snippets_no_label = snippet_table[
        snippet_table[orcai_parameter["calls"]].sum(axis=1) <= 0.0000001
    ]
    p_no_label_before = np.around(
        100 * len(snippets_no_label) / snippet_table.shape[0], 2
    )
    msgr.info(
        f"Percentage of snippets containing no label before selection: {str(p_no_label_before)} %"
    )

    # removing rows from extracted_snippets where there is no label present
    msgr.info(
        f"removing {np.around(orcai_parameter['snippets']['fraction_removal'] * 100, 2)}% of snippets without label"
    )

    indices_to_drop = rng.choice(
        snippets_no_label.index,
        size=int(
            orcai_parameter["snippets"]["fraction_removal"] * len(snippets_no_label)
        ),
        replace=False,
    )
    snippet_table = snippet_table.drop(indices_to_drop, axis=0)

    snippets_no_label = snippet_table[
        snippet_table[orcai_parameter["calls"]].sum(axis=1) <= 0.0000001
    ]
    p_no_label_after = np.around(
        100 * len(snippets_no_label) / snippet_table.shape[0], 2
    )

    msgr.info(
        f"Percentage of snippets containing no label after selection: {str(p_no_label_after)} %"
    )
    snippet_table = snippet_table.reset_index(drop=True)

    msgr.info("Number of train, val, test snippets:", indent=1)
    msgr.info(snippet_table.groupby("data_type").size(), indent=-1)

    return snippet_table


def create_tvt_snippet_tables(
    output_dir: Path | str,
    snippet_table: (Path | str) | pd.DataFrame | None = None,
    orcai_parameter: Path | str = files("orcAI.defaults").joinpath(
        "default_orcai_parameter.json"
    ),
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> None:
    """Creates snippet tables for training, validation and test datasets and saves them to disk

    Parameters
    ----------
    output_dir : (Path | str)
        Path to the output directory
    snippet_table : (Path | str) | pd.DataFrame | None
        Path to the snippet table csv or the snippet table itself. None if the snippet table should be read from output_dir/all_snippets.csv.gz
    orcai_parameter : dict | (Path | str)
        Dict containing OrcAi parameter or path to json containing the same, by default files("orcAI.defaults").joinpath("default_orcai_parameter.json")
    verbosity : int
        Verbosity level [0, 1, 2]
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.


    Returns
    -------
    None. Writes train, val and test snippet tables to disk
    """
    if msgr is None:
        msgr = Messenger(
            verbosity=verbosity,
            title="Creating train, validation and test snippet tables",
        )

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    msgr.part("Reading snippet table")

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)

    if snippet_table is None:
        snippet_table = Path(output_dir).joinpath("all_snippets.csv.gz")
    if isinstance(snippet_table, (Path | str)):
        snippet_table = pd.read_csv(snippet_table)

    all_snippet_stats = _compute_snippet_stats(
        snippet_table, for_calls=orcai_parameter["calls"]
    )
    all_snippet_stats_duration = all_snippet_stats.filter(
        regex=".*(?<!_ef)$", axis=1
    ).map(seconds_to_hms)
    msgr.info("Snippet stats [HMS]:", indent=1)
    msgr.info(all_snippet_stats_duration, indent=-1)
    all_snippet_stats_duration.to_csv(
        output_dir.joinpath("all_snippet_stats_duration.csv"), index=True
    )

    rng = np.random.default_rng(
        seed=[SEED_ID_FILTER_SNIPPET_TABLE, orcai_parameter["seed"]]
    )
    snippet_table_filtered = _filter_snippet_table(
        snippet_table,
        orcai_parameter=orcai_parameter,
        rng=rng,
        msgr=msgr,
    )

    snippets = []
    for i, itype in enumerate(["train", "val", "test"]):
        n_snippets = (
            (orcai_parameter["model"][f"n_batch_{itype}"])
            * orcai_parameter["model"]["batch_size"]
        )
        msgr.info(
            f"Extracting {orcai_parameter['model'][f'n_batch_{itype}']} batches of {orcai_parameter['model']['batch_size']} random {itype} snippets ({n_snippets} snippets)"
        )
        snippet_table_i = snippet_table_filtered[
            snippet_table_filtered["data_type"] == itype
        ]
        if len(snippet_table_i) < n_snippets:
            msgr.error(
                f"Number of {itype} snippets ({n_snippets}) larger than available snippets ({len(snippet_table_i)})."
            )
            msgr.error("Skipping.")
            continue

        snippets.append(
            snippet_table_i.sample(n=n_snippets, replace=False, random_state=rng)
        )

        snippets[i][["recording_data_dir", "row_start", "row_stop"]].to_csv(
            output_dir.joinpath(f"{itype}.csv.gz"), compression="gzip", index=False
        )

    selected_snippet_stats = _compute_snippet_stats(
        pd.concat(snippets, ignore_index=True), for_calls=orcai_parameter["calls"]
    )
    selected_snippet_stats_duration = selected_snippet_stats.filter(
        regex=".*(?<!_ef)$", axis=1
    ).map(seconds_to_hms)
    msgr.info("Snippet stats for train, val and test datasets [HMS]:", indent=1)
    msgr.info(selected_snippet_stats_duration, indent=-1)
    selected_snippet_stats_duration.to_csv(
        output_dir.joinpath("selected_snippet_stats_duration.csv"), index=True
    )

    msgr.success("Train, val and test snippet tables created and saved to disk")

    return


def create_tvt_data(
    tvt_dir: Path | str,
    orcai_parameter: dict | (Path | str) = files("orcAI.defaults").joinpath(
        "default_orcai_parameter.json"
    ),
    overwrite: bool = False,
    data_compression: str | None = "GZIP",
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> dict[str, tf.data.Dataset]:
    """Creates train, validation and test datasets from snippet tables and saves them to disk

    Parameters
    ----------
    tvt_dir : (Path | str)
        Path to the directory containing the training, validation and test snippet tables
    orcai_parameter : dict | (Path | str)
        Dict containing model specifications or path to json containing the same, by default files("orcAI.defaults").joinpath("default_orcai_parameter.json")
    overwrite : bool
        Overwrite existing datasets
    data_compression: str | None
        Compression for data files. Accepts "GZIP" or "NONE".
    verbosity : int
        Verbosity level [0, 1, 2]
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.


    Returns
    -------
    dataset : dict[str, tf.data.Dataset]
        Dictionary containing train, val and test datasets
        Writes train, val and test datasets to tvt_dir
    """
    if msgr is None:
        msgr = Messenger(
            verbosity=verbosity,
            title="Creating train, validation and test datasets",
        )

    msgr.part("Reading in snippet tables and generating loaders")

    data_types = ["train", "val", "test"]

    dataset_paths = {itype: Path(tvt_dir, f"{itype}_dataset") for itype in data_types}

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)

    csv_paths = {itype: Path(tvt_dir, f"{itype}.csv.gz") for itype in data_types}

    loader = {
        key: DataLoader.from_csv(
            path,
            len(orcai_parameter["model"]["filters"]),
            shuffle=True,
            rng=np.random.default_rng(
                seed=[SEED_ID_CREATE_DATALOADER.get(key, 0), orcai_parameter["seed"]]
            ),
        )
        for key, path in csv_paths.items()
    }

    msgr.info("Data shape:", indent=1)
    spectrogram_sample, label_sample = loader[data_types[0]][0]
    msgr.info(f"Input spectrogram batch shape: {spectrogram_sample.shape}")
    msgr.info(f"Input label batch shape: {label_sample.shape}", indent=-1)

    msgr.part("Creating test, validation and training datasets")
    dataset = {}
    for itype in data_types:
        dataset[itype] = tf.data.Dataset.from_generator(
            loader[itype].__iter__,
            output_signature=(
                tf.TensorSpec(
                    shape=spectrogram_sample.shape,
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=label_sample.shape,
                    dtype=tf.float32,
                ),
            ),
        )
        dataset[itype] = dataset[itype].prefetch(tf.data.experimental.AUTOTUNE)
        msgr.info(f"{itype.capitalize()} dataset created. Length {len(loader[itype])}.")

    msgr.part("Saving datasets to disk")

    for itype in data_types:
        try:
            save_dataset(
                dataset[itype],
                path=dataset_paths[itype],
                overwrite=overwrite,
                compression=data_compression,
            )
        except FileExistsError as _:
            msgr.warning(
                f"File {dataset_paths[itype]} already exists. Skipping. Set overwrite=True to overwrite."
            )
        msgr.print_directory_size(dataset_paths[itype])

    write_json(
        {
            "spectrogram": spectrogram_sample.shape.as_list(),
            "labels": label_sample.shape.as_list(),
        },
        Path(tvt_dir, "dataset_shapes.json"),
    )
    msgr.success("Train, validation and test datasets created and saved to disk")

    return dataset
