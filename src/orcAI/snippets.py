import time
from pathlib import Path
from importlib.resources import files
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
import tensorflow as tf


from orcAI.auxiliary import (
    Messenger,
    read_json,
    seconds_to_hms,
    resolve_recording_data_dir,
)
from orcAI.load import load_data_from_snippet_csv, data_generator


def _make_snippet_table(
    recording_dir: str | Path,
    orcai_parameter: dict,
    msgr: Messenger = Messenger(verbosity=2),
):
    """Generates times for snippets to be extracted from labels

    returns pd.DataFrame with recording, data_type, start, stop and duration for each label

    Parameters
    ----------
    recording_dir : str | Path
        Path to the recording data directory
    orcai_parameter : dict
        dict containing orcai parameter
    msgr : Messenger
        Messenger object for messages

    Returns
    -------
    pd.DataFrame
        snippet table with columns recording, data_type, row_start, row_stop and call names
    Int
        recording duration
    """
    recording = Path(recording_dir).stem
    label_zarr_path = Path(recording_dir).joinpath("labels", "labels.zarr")
    label_list_path = Path(recording_dir).joinpath("labels", "label_list.json")
    spectrogram_times_path = Path(recording_dir).joinpath("spectrogram", "times.json")

    try:
        spectrogram_times = read_json(spectrogram_times_path)
    except FileNotFoundError as e:
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

    # zarr open doesn't work with try/except blocks
    if label_zarr_path.exists() & label_list_path.exists():
        label_filepointer = zarr.open(label_zarr_path, mode="r")
        label_list = read_json(label_list_path)
        label_names = list(label_list.keys())
    else:
        msgr.warning(f"File not found: {label_zarr_path}")
        return (None, recording_duration, n_segments, recording, "missing label files")

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
                t_start = np.random.uniform(low=t_min, high=t_max, size=1)[0]

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


def _compute_snippet_stats(snippet_table: pd.DataFrame, for_calls: list):
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
    recording_table_path: str | Path,
    recording_data_dir: str | Path,
    orcai_parameter: dict | (str | Path) = files("orcAI.defaults").joinpath(
        "default_orcai_parameter.json"
    ),
    verbosity: int = 2,
):
    """Generates snippet table for all recordings in recording_table and saves it to disk

    Parameters
    ----------
    recording_table_path : (str | Path)
        Path to the recording table
    recording_data_dir : (str | Path)
        Path to the recording data directory
    orcai_parameter : dict | (str | Path)
        Dict containing OrcAI parameter or path to json containing the same, by default files("orcAI.defaults").joinpath("default_orcai_parameter.json")
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug

    Returns
    -------
    None. Writes snippet table to disk
    """
    msgr = Messenger(verbosity=verbosity)
    msgr.part("Making snippet table")

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)

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
        f"Created snippet table for {len(snippet_table['recording'].unique())} recordings. "
        + f"Total recording duration: {seconds_to_hms(np.sum(recording_lengths))}. "
        + f"Total number of snippets: {len(snippet_table)}. "
        + f"Total number of segments: {np.sum(segments)}"
    )

    msgr.info(f"Creating snippet table failed for {len(failed)} recordings.")
    msgr.info(failed_table.groupby("reason").size())

    failed_table.to_csv(
        Path(recording_data_dir).joinpath("failed_snippets.csv"),
        index=False,
    )

    snippet_table.to_csv(
        Path(recording_data_dir).joinpath("all_snippets.csv.gz"),
        compression="gzip",
        index=False,
    )
    msgr.success(
        f"Snippet table saved to {Path(recording_data_dir).joinpath('all_snippets.csv.gz')}"
    )

    return


def _filter_snippet_table(
    snippet_table: pd.DataFrame,
    snippet_parameter: dict,
    label_calls: dict,
    msgr: Messenger = Messenger(verbosity=2),
):
    """Filters snippet table based on snippet parameter and label calls

    Parameters
    ----------
    snippet_table : pd.DataFrame
        snippet_table with columns recording, data_type, row_start, row_stop and call names
    snippet_parameter : dict
        dict containing snippet parameter
    label_calls : dict
        dict containing calls for labeling
    msgr : Messenger
        Messenger object for messages

    Returns
    -------
    pd.DataFrame
        filtered snippet table
    """
    msgr.part("Filtering snippet table")

    snippet_stats = _compute_snippet_stats(snippet_table, for_calls=label_calls)
    snippet_stats_duration = snippet_stats.filter(regex=".*(?<!_ef)$", axis=1).map(
        seconds_to_hms
    )
    msgr.info("Snippet stats [HMS]:")
    msgr.info(snippet_stats_duration)

    snippets_no_label = snippet_table[
        snippet_table[label_calls].sum(axis=1) <= 0.0000001
    ]
    p_no_label_before = np.around(
        100 * len(snippets_no_label) / snippet_table.shape[0], 2
    )
    msgr.info(
        f"Percentage of snippets containing no label before selection: {str(p_no_label_before)} %"
    )

    # removing rows from extracted_snippets where there is no label present
    msgr.info(
        f"removing {np.around(snippet_parameter['fraction_removal'] * 100, 2)}% of snippets without label"
    )
    indices_to_drop = np.random.choice(
        snippets_no_label.index,
        size=int(snippet_parameter["fraction_removal"] * len(snippets_no_label)),
        replace=False,
    )
    snippet_table = snippet_table.drop(indices_to_drop, axis=0)

    snippets_no_label = snippet_table[
        snippet_table[label_calls].sum(axis=1) <= 0.0000001
    ]
    p_no_label_after = np.around(
        100 * len(snippets_no_label) / snippet_table.shape[0], 2
    )

    msgr.info(
        f"Percentage of snippets containing no label after selection: {str(p_no_label_after)} %"
    )
    snippet_table = snippet_table.reset_index(drop=True)

    msgr.info("Number of train, val, test snippets:")
    msgr.info(snippet_table.groupby("data_type").size())

    return snippet_table


def create_tvt_snippet_tables(
    recording_data_dir,
    output_dir,
    snippet_table=None,
    orcai_parameter=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    verbosity=2,
):
    """Creates snippet tables for training, validation and test datasets and saves them to disk

    Parameters
    ----------
    recording_data_dir : (str | Path)
        Path to the recording data directory
    output_dir : (str | Path)
        Path to the output directory
    snippet_table : (str | Path) | pd.DataFrame | None
        Path to the snippet table csv or the snippet table itself. None if the snippet table should be read from recording_data_dir/all_snippets.csv.gz
    orcai_parameter : dict | (str | Path)
        Dict containing OrcAi parameter or path to json containing the same, by default files("orcAI.defaults").joinpath("default_orcai_parameter.json")
    verbosity : int
        Verbosity level [0, 1, 2]

    Returns
    -------
    None. Writes train, val and test snippet tables to disk
    """
    msgr = Messenger(verbosity=verbosity)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    msgr.part("Extracting snippets")

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)
    snippet_parameter = orcai_parameter["snippets"]
    label_calls = orcai_parameter["calls"]

    if snippet_table is None:
        snippet_table = Path(recording_data_dir).joinpath("all_snippets.csv.gz")
    if isinstance(snippet_table, (Path | str)):
        snippet_table = pd.read_csv(snippet_table)

    msgr.info("Filtering snippet table")
    snippet_table_filtered = _filter_snippet_table(
        snippet_table,
        snippet_parameter=snippet_parameter,
        label_calls=label_calls,
        msgr=msgr,
    )

    for itype in ["train", "val", "test"]:
        n_snippets = snippet_parameter[f"n_{itype}"]
        msgr.info(f"Extracting {n_snippets} {itype} snippets")
        snippet_table_i = snippet_table_filtered[
            snippet_table_filtered["data_type"] == itype
        ]
        try:
            snippets_itype = snippet_table_i.sample(n=n_snippets, replace=False)
        except ValueError as e:
            msgr.error(
                f"Number of {itype} snippets ({n_snippets}) larger than available snippets ({len(snippet_table_i)})."
            )
            raise e
        snippets_itype[["recording_data_dir", "row_start", "row_stop"]].to_csv(
            Path(output_dir, f"{itype}.csv.gz"), compression="gzip", index=False
        )

    msgr.success("Train, val and test snippet tables created and saved to disk")

    return


def create_tvt_data(
    tvt_dir,
    orcai_parameter=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    verbosity=2,
):
    """Creates train, validation and test datasets from snippet tables and saves them to disk

    Parameters
    ----------
    tvt_dir : (str | Path)
        Path to the directory containing the training, validation and test snippet tables
    orcai_parameter : dict | (str | Path)
        Dict containing model specifications or path to json containing the same, by default files("orcAI.defaults").joinpath("default_orcai_parameter.json")
    verbosity : int
        Verbosity level [0, 1, 2]

    Returns
    -------
    None. Writes train, val and test datasets to tvt_dir
    """
    msgr = Messenger(verbosity=verbosity)
    msgr.part("Creating train, validation and test data")
    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)
    model_parameter = orcai_parameter["model"]

    csv_paths = [Path(tvt_dir, f"{itype}.csv.gz") for itype in ["train", "val", "test"]]

    msgr.info("Reading in dataframes with snippets and generating loaders", indent=1)
    start_time = time.time()
    loaders, spectrogram_chunk_shape, label_chunk_shape = load_data_from_snippet_csv(
        csv_paths, model_parameter, msgr=msgr
    )
    msgr.info(
        f"Dataframes read and loaders generated in {seconds_to_hms(time.time() - start_time)}"
    )
    msgr.info(f"Spectrogram chunk shape: {spectrogram_chunk_shape}")
    msgr.info(f"Original label chunk shape: {label_chunk_shape}")
    msgr.print_memory_usage(indent=-1)

    msgr.info("Data characteristics:", indent=1)
    start_time = time.time()
    spectrogram_batch, label_batch = loaders["train"][0]
    msgr.info(f"Data loading time per batch: {time.time() - start_time:.2f} seconds")
    msgr.info(f"Input spectrogram batch shape: {spectrogram_batch.shape}")
    msgr.info(f"Input label batch shape: {label_batch.shape}", indent=-1)

    msgr.info("Creating test, validation and training datasets", indent=1)
    start_time = time.time()
    dataset = {}
    for itype in ["train", "val", "test"]:
        dataset[itype] = tf.data.Dataset.from_generator(
            lambda: data_generator(loaders[itype]),
            output_signature=(
                tf.TensorSpec(
                    shape=(
                        spectrogram_batch.shape[1],
                        spectrogram_batch.shape[2],
                        1,
                    ),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(label_batch.shape[1], label_batch.shape[2]),
                    dtype=tf.float32,
                ),
            ),
        )
        msgr.info(f"{itype.capitalize()} dataset created")
    msgr.info(f"Datasets created in {seconds_to_hms(time.time() - start_time)}")
    msgr.print_memory_usage(indent=-1)

    msgr.info("Saving datasets to disk", indent=1)

    # TODO: test saving
    for itype in ["train", "val", "test"]:
        start_time = time.time()
        dataset_path = Path(tvt_dir, f"{itype}_dataset")
        dataset[itype].save(
            path=str(dataset_path)  # TODO: check path before trying to save
        )  # deadlocks silently on error https://github.com/tensorflow/tensorflow/issues/61736
        msgr.info(
            f"{itype.capitalize()} dataset saved to disk in {seconds_to_hms(time.time() - start_time)}"
        )
        msgr.print_directory_size(dataset_path)

    msgr.success("Train, validation and test datasets created and saved to disk")

    return
