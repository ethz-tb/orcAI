import time
from pathlib import Path
from importlib.resources import files
import numpy as np
import pandas as pd
from click import progressbar
import zarr
import tensorflow as tf


from orcAI.auxiliary import (
    Messenger,
    read_json,
    seconds_to_hms,
    resolve_recording_data_dir,
    recording_table_show_func,
)
from orcAI.load import load_data_from_snippet_csv, data_generator


def _make_snippet_table(
    recording_data_dir,
    snippet_parameter=files("orcAI.defaults").joinpath(
        "default_snippet_parameter.json"
    ),
    model_parameter=files("orcAI.defaults").joinpath("default_model_parameter.json"),
    msgr=Messenger(verbosity=2),
):
    """Generates times for snippets to be extracted from labels

    returns pd.DataFrame with recording, data_type, start, stop and duration for each label

    Parameters
    ----------
    recording_data_dir : str
        Path to the recording data directory
    snippet_parameter : (str | Path)
        Path to a JSON file containing segment specifications, by default files("orcAI.defaults").joinpath("default_snippet_parameter.json")
    model_parameter : (str | Path)
        Path to a JSON file containing model specifications, by default files("orcAI.defaults").joinpath("default_model_parameter.json")
    msgr : Messenger
        Messenger object for messages
    Returns
    -------
    pd.DataFrame
        snippet table with columns recording, data_type, row_start, row_stop and call names
    Int
        recording duration
    """
    recording = Path(recording_data_dir).stem
    label_zarr_path = Path(recording_data_dir).joinpath("labels", "labels.zarr")
    label_list_path = Path(recording_data_dir).joinpath("labels", "label_list.json")
    spectrogram_times_path = Path(recording_data_dir).joinpath(
        "spectrogram", "times.json"
    )

    try:
        spectrogram_times = read_json(spectrogram_times_path)
    except FileNotFoundError as e:
        msgr.error(f"File not found: {spectrogram_times_path}")
        msgr.error("Did you create the spectrogram?")
        raise

    if isinstance(snippet_parameter, (Path | str)):
        snippet_parameter = read_json(snippet_parameter)

    recording_duration = spectrogram_times["max"]
    n_segments = int(recording_duration // snippet_parameter["length"])
    if n_segments <= 0:
        msgr.warning(
            f"Duration of recording ({spectrogram_times['max']}) is shorter than segment length ({snippet_parameter['length']}). Skipping recording."
        )
        return pd.DataFrame()

    # zarr open doesn't work with try/except blocks
    if label_zarr_path.exists() & label_list_path.exists():
        label_filepointer = zarr.open(label_zarr_path, mode="r")
        label_list = read_json(label_list_path)
        label_names = list(label_list.keys())
    else:
        msgr.error(f"File not found: {label_zarr_path}")
        msgr.error("Wrong path? Did you create the labels?")
        raise FileNotFoundError

    if isinstance(model_parameter, (Path | str)):
        model_parameter = read_json(model_parameter)

    times = np.linspace(
        spectrogram_times["min"],
        spectrogram_times["max"],
        spectrogram_times["length"],
    )
    delta_t = times[1] - times[0]
    n_filters = len(model_parameter["filters"])
    i_duration = int(
        (2**n_filters) * ((snippet_parameter["duration"] / delta_t) // (2**n_filters))
    )  # to make time axis divisible by 2 ** n_filters
    msgr.info(f"i_duration: {i_duration}")  # TODO: clarify
    snippet_table_raw = []
    for i_segment in range(n_segments):  # iterate over all segments
        msgr.info("Segment", i_segment + 1, "of", n_segments)
        slice = (0, 0)
        for type in list(["train", "val", "test"]):  # iterate over type of snippet
            slice = (slice[1], slice[1] + model_parameter[type])
            t_min = (i_segment + slice[0]) * snippet_parameter["length"]
            for j in range(
                int(
                    model_parameter[type]
                    * snippet_parameter["length"]
                    * snippet_parameter["per_sec"]
                )
            ):  # iterate over number of snippets per segment and type
                t_max = (i_segment + slice[1]) * snippet_parameter[
                    "length"
                ] - snippet_parameter["duration"]
                t_start = np.random.uniform(low=t_min, high=t_max, size=1)[0]

                # Find the max index where entries are smaller than t_start
                index_t_start = np.searchsorted(times, t_start, side="left") - 1
                # Find the min index where entries are smaller or equal to t_stop
                index_t_stop = index_t_start + i_duration
                label_chunk = label_filepointer[index_t_start:index_t_stop, :]
                label_duration_snippet = label_chunk.sum(axis=0) * delta_t
                label_duration_snippet[label_duration_snippet < 0] = np.nan
                snippet_table_raw += [
                    list(
                        [
                            recording,
                            recording_data_dir,
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
    return (snippet_table, recording_duration, n_segments)


def _compute_snippet_stats(snippet_table, for_calls):
    """Compute snippet stats for calls

    Parameters
    ----------
    snippet_table : pd.DataFrame
        snippet_table with columns recording, data_type, row_start, row_stop and call names
    for_calls : list
        list of call names to compute stats for
    """

    snippet_stats = snippet_table.groupby("data_type")[for_calls].sum().T
    snippet_stats["total"] = snippet_stats.sum(axis=1)

    equalizing_factors = snippet_stats.apply(lambda x: 1 / x * x.max(), axis=0)
    equalizing_factors.columns = equalizing_factors.columns + "_ef"

    return pd.merge(
        snippet_stats, equalizing_factors, left_index=True, right_index=True
    )


def create_snippet_table(
    recording_table_path,
    recording_data_dir,
    snippet_parameter=files("orcAI.defaults").joinpath(
        "default_snippet_parameter.json"
    ),
    model_parameter=files("orcAI.defaults").joinpath("default_model_parameter.json"),
    label_calls=files("orcAI.defaults").joinpath("default_calls.json"),
    save_snippet_table=True,
    verbosity=2,
):
    """Generates snippet table for all recordings"""
    msgr = Messenger(verbosity=verbosity)
    msgr.part("Making snippet table")

    if isinstance(label_calls, (Path | str)):
        label_calls = read_json(label_calls)

    if isinstance(snippet_parameter, (Path | str)):
        snippet_parameter = read_json(snippet_parameter)

    recording_table = pd.read_csv(recording_table_path)
    recording_table["recording_data_dir"] = recording_table.apply(
        lambda row: resolve_recording_data_dir(row["recording"], recording_data_dir),
        axis=1,
    )

    missing_recording_data_dir = pd.isna(recording_table["recording_data_dir"])
    if any(missing_recording_data_dir):
        msgr.warning(
            f"Missing recording data directories for {sum(missing_recording_data_dir)} recordings. Skipping these recordings."
        )
        recording_table = recording_table[~missing_recording_data_dir]

    recording_lengths = []
    segments = []
    all_snippet_tables = []

    with progressbar(
        recording_table.index,
        label="Making snippet tables",
        item_show_func=lambda index: recording_table_show_func(
            index,
            recording_table,
        ),
    ) as recording_indices:
        for i in recording_indices:
            snippet_table, recording_length, n_segments = _make_snippet_table(
                recording_table.loc[i, "recording_data_dir"],
                snippet_parameter=snippet_parameter,
                model_parameter=model_parameter,
                msgr=Messenger(verbosity=0),
            )
            all_snippet_tables.append(snippet_table)
            recording_lengths.append(recording_length)
            segments.append(n_segments)

    snippet_table = pd.concat(all_snippet_tables).reset_index(drop=True)
    msgr.info(
        f"Created snippet table for {len(recording_table)} recordings. "
        + f"Total recording duration: {seconds_to_hms(np.sum(recording_lengths))}. "
        + f"Total number of snippets: {len(snippet_table)}. "
        + f"Total number of segments: {np.sum(segments)}"
    )

    if save_snippet_table:
        snippet_table.to_csv(
            Path(recording_data_dir).joinpath("all_snippets.csv.gz"),
            compression="gzip",
            index=False,
        )
        msgr.success(
            f"Snippet table saved to {Path(recording_data_dir).joinpath('all_snippets.csv.gz')}"
        )

    return snippet_table


def _filter_snippet_table(
    snippet_table,
    snippet_parameter=files("orcAI.defaults").joinpath(
        "default_snippet_parameter.json"
    ),
    label_calls=files("orcAI.defaults").joinpath("default_calls.json"),
    msgr=Messenger(verbosity=2),
):
    msgr.part("Filtering snippet table")

    snippet_stats = _compute_snippet_stats(snippet_table, for_calls=label_calls)
    snippet_stats_duration = snippet_stats.filter(regex=".*(?<!_ef)$", axis=1).map(
        seconds_to_hms
    )
    msgr.info("Snippet stats [HMS]:")
    msgr.info(snippet_stats_duration)

    indices_no_label = np.where(snippet_table[label_calls].sum(axis=1) <= 0.0000001)[0]
    p_no_label_before = np.around(
        100 * len(indices_no_label) / snippet_table.shape[0], 2
    )
    msgr.info(
        f"Percentage of snippets containing no label before selection: {str(p_no_label_before)} %"
    )

    # removing rows from extracted_snippets where there is no label present
    msgr.info(
        f"removing {np.around(snippet_parameter['fraction_removal'] * 100, 2)}% of snippets without label"
    )
    indices_to_drop = np.random.choice(
        indices_no_label,
        size=int(snippet_parameter["fraction_removal"] * len(indices_no_label)),
        replace=False,
    )
    # indices_to_drop.sort()
    # snippet_table = snippet_table.reset_index(drop=True)
    # TODO: this will change the indices essentially dropping different rows than selected. still random, i guess?

    snippet_table = snippet_table[~snippet_table.index.isin(indices_to_drop)]
    indices_no_label = np.where(snippet_table[label_calls].sum(axis=1) <= 0.0000001)[0]
    p_no_label_after = np.around(
        100 * len(indices_no_label) / snippet_table.shape[0], 2
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
    snippet_parameter=files("orcAI.defaults").joinpath(
        "default_snippet_parameter.json"
    ),
    label_calls=files("orcAI.defaults").joinpath("default_calls.json"),
    verbosity=2,
):
    """Creates snippet tables and saves them to disk"""
    msgr = Messenger(verbosity=verbosity)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    msgr.part("Extracting snippets")

    if isinstance(snippet_parameter, (Path | str)):
        msgr.info(f"Reading snippet parameter from {snippet_parameter}")
        snippet_parameter = read_json(snippet_parameter)

    if snippet_table is None:
        snippet_table = Path(recording_data_dir).joinpath("all_snippets.csv.gz")
    if isinstance(snippet_table, (Path | str)):
        snippet_table = pd.read_csv(snippet_table)

    if isinstance(label_calls, (Path | str)):
        label_calls = read_json(label_calls)

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
    model_parameter=files("orcAI.defaults").joinpath("default_model_parameter.json"),
    verbosity=2,
):
    msgr = Messenger(verbosity=verbosity)
    msgr.part("Creating train, validation and test data")
    if isinstance(model_parameter, (Path | str)):
        model_parameter = read_json(model_parameter)

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
            path=str(dataset_path)
        )  # deadlocks silently on error https://github.com/tensorflow/tensorflow/issues/61736
        msgr.info(
            f"{itype.capitalize()} dataset saved to disk in {seconds_to_hms(time.time() - start_time)}"
        )
        msgr.print_directory_size(dataset_path)

    msgr.success("Train, validation and test datasets created and saved to disk")

    return
