from importlib.resources import files
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from tqdm import tqdm

from orcAI.auxiliary import Messenger, find_consecutive_ones
from orcAI.io import load_orcai_model, read_json
from orcAI.spectrogram import make_spectrogram


def _check_duration(
    calls: pd.DataFrame,
    call_duration_limits: dict[str : tuple[float | None, float | None]],
    delta_t: float,
    label_suffix: str = "*",
) -> str:
    """
    Checks duration of calls against call duration limits.

    Parameter
    ----------
    x : pd.DataFrame
        DataFrame with calls of a single label.
    call_duration_limits : dict
        Dictionary with call duration limits for each label.
    delta_t : float
        Time step duration in seconds.
    label_suffix : str
         Suffix that was added to label names during prediction.


    Returns
    -------
    out : str
        str "keep", "too long", or "too short"

    """
    label = calls["label"].replace(f"{label_suffix}", "")

    if label in call_duration_limits:
        min_duration, max_duration = call_duration_limits[label]
        if min_duration is None:
            min_duration = 0
        if max_duration is None:
            max_duration = np.inf
    elif "default" in call_duration_limits:
        min_duration, max_duration = call_duration_limits["default"]
        if min_duration is None:
            min_duration = 0
        if max_duration is None:
            max_duration = np.inf
    else:
        min_duration = 0
        max_duration = np.inf

    if calls["duration"] * delta_t < min_duration:
        out = "too short"
    elif calls["duration"] * delta_t > max_duration:
        out = "too long"
    else:
        out = "keep"

    return out


def filter_predictions(
    predicted_labels: pd.DataFrame,
    delta_t: float,
    call_duration_limits: (Path | str) | dict = files("orcAI.defaults").joinpath(
        "default_call_duration_limits.json"
    ),
    label_suffix: str = "*",
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> pd.DataFrame:
    """
    Filter predictions based on duration.

    Parameter
    ----------
    predicted_labels : pd.DataFrame
        DataFrame with predicted labels or path to a file with predicted labels.
    output_file : (Path | str) | "default" | None
        Path to the output file or "default" to save in the same directory as the predicted labels file. None to not save predictions to disk.
    call_duration_limits : (Path | str) | dict
        Path to a JSON file containing a dictionary with call duration limits.
    label_suffix : str
        Suffix that was added to label names during prediction.
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    msgr : Messenger
        Messenger object for logging. If None a new Messenger object is created.

    Returns
    -------
    predicted_labels_duration_ok : pd.DataFrame
        DataFrame with predicted labels after filtering based
        on duration.

    """

    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Filtering predictions")
    msgr.part("Filtering predictions")

    predicted_labels["duration"] = predicted_labels["stop"] - predicted_labels["start"]

    if msgr.verbosity >= 3:
        predicted_labels["duration_s"] = predicted_labels["duration"] * delta_t
        msgr.debug("Call durations:")
        msgr.debug(
            predicted_labels[["label", "duration_s"]].groupby("label").describe()
        )
    if isinstance(call_duration_limits, (Path | str)):
        call_duration_limits = read_json(call_duration_limits)
    msgr.debug("Call duration limits:")
    msgr.debug(call_duration_limits)

    # Filter calls based on duration
    msgr.part("Filtering calls based on duration")
    predicted_labels["duration_ok"] = predicted_labels.apply(
        lambda x: _check_duration(x, call_duration_limits, delta_t, label_suffix),
        axis=1,
    )

    duration_check_summary = (
        predicted_labels[["label", "duration_ok"]]
        .groupby("label")
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
    )

    msgr.debug("Summary of kept and discarded calls based on duration:")
    msgr.debug(duration_check_summary)
    try:
        n_too_long = duration_check_summary["too long"].sum()
    except KeyError:
        n_too_long = 0
    try:
        n_too_short = duration_check_summary["too short"].sum()
    except KeyError:
        n_too_short = 0

    msgr.info(
        f"Discarding {n_too_long + n_too_short} calls based on duration (too short: {n_too_short}, too long: {n_too_long})"
    )

    predicted_labels_duration_ok = predicted_labels[
        predicted_labels["duration_ok"] == "keep"
    ]

    msgr.success("Filtering predictions finished.")

    return predicted_labels_duration_ok


def filter_predictions_file(
    predicted_labels: Path | str,
    output_file: Path | str = "default",
    overwrite: bool = False,
    call_duration_limits: (Path | str) | dict = files("orcAI.defaults").joinpath(
        "default_call_duration_limits.json"
    ),
    label_suffix: str = "*",
    verbosity: int = 2,
    msgr: Messenger | None = None,
):
    """
    Filter predictions file based on duration.

    Parameter
    ----------
    predicted_labels : (Path | str)
        Path to a file with predicted labels.
    output_file : (Path | str) | "default"
        Path to the output file or "default" to save in the same directory as the predicted labels file.
    overwrite : bool
        Overwrite the output file if it exists.
    call_duration_limits : (Path | str) | dict
        Path to a JSON file containing a dictionary with call duration limits.
    label_suffix : str
        Suffix that was added to label names during prediction.
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    msgr : Messenger
        Messenger object for logging. If None a new Messenger object is created.

    Returns
    -------
    predicted_labels_duration_ok : pd.DataFrame
        DataFrame with predicted labels after filtering based
        on duration.

    Raises
    ------
    ValueError
        If the output file already exists.

    """
    if output_file == "default":
        filename = Path(predicted_labels).stem + "_filtered.txt"
        output_file = Path(predicted_labels).with_name(filename)
    else:
        output_file = Path(output_file)
    msgr.info(f"Output file: {output_file}")

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Annotation file already exists: {output_file}")

    predicted_labels = pd.read_csv(predicted_labels, sep="\t", encoding="utf-8")

    predicted_labels_duration_ok = filter_predictions(
        predicted_labels=predicted_labels,
        delta_t=1,
        call_duration_limits=call_duration_limits,
        label_suffix=label_suffix,
        verbosity=verbosity,
        msgr=msgr,
    )

    save_predictions(
        predicted_labels=predicted_labels_duration_ok,
        output_path=output_file,
        delta_t=1,
        msgr=msgr,
    )
    return


def compute_aggregated_predictions(
    recording_path: Path,
    spectrogram: np.ndarray,
    model: keras.Model,
    orcai_parameter: dict,
    shape: dict,
    msgr: Messenger = Messenger(verbosity=0),
    progressbar: tqdm = None,
) -> tuple[np.ndarray, np.ndarray]:
    snippet_length = shape["input_shape"][0]  # Time steps in a single snippet

    shift = snippet_length // 2  # Shift time steps for overlapping windows
    time_steps_per_output_step = 2 ** len(orcai_parameter["model"]["filters"])
    prediction_length = (
        snippet_length // time_steps_per_output_step
    )  # Output time steps per prediction

    # Step 1: Create overlapping spectrogram snippets
    num_snippets = (spectrogram.shape[0] - snippet_length) // shift + 1
    msgr.info(f"slicing into {num_snippets} snippets for prediction")

    snippets = np.array(
        [
            spectrogram[i * shift : i * shift + snippet_length]
            for i in range(num_snippets)
        ]
    )  # Shape: (num_snippets, 736, 171)

    # Step 2: Model predictions for all snippets
    msgr.info("Prediction of snippets")
    snippets = snippets[..., np.newaxis]  # Shape: (num_snippets, 736, 171, 1)
    predictions = model.predict(
        snippets, verbose=0 if msgr.verbosity < 2 else 1
    )  # Shape: (num_snippets, 46, 7)

    # Step 3: Initialize arrays for aggregating predictions
    msgr.info("Aggregating predictions")
    if progressbar:
        progressbar.set_description(f"{recording_path.stem} - Aggregating predictions")
        progressbar.refresh()

    total_time_steps = spectrogram.shape[0] // time_steps_per_output_step
    aggregated_predictions = np.zeros((total_time_steps, shape["num_labels"]))
    overlap_count = np.zeros(
        total_time_steps
    )  # To track the number of overlaps per time step

    # Step 4: Overlay predictions
    for i, prediction in enumerate(predictions):
        start = i * (
            shift // time_steps_per_output_step
        )  # Start index in aggregated predictions
        end = start + prediction_length  # End index
        aggregated_predictions[start:end] += prediction  # Add predictions
        overlap_count[start:end] += 1  # Track overlaps

    # Step 5: Average the overlapping predictions
    valid_mask = overlap_count > 0  # removes zeros at end
    aggregated_predictions[valid_mask] /= overlap_count[valid_mask, np.newaxis]

    return aggregated_predictions, overlap_count


def compute_binary_predictions(
    aggregated_predictions: np.ndarray,
    overlap_count: np.ndarray,
    calls: list[str],
    threshold: float = 0.5,
) -> tuple[list[int], list[int], list[str]]:
    adjusted_threshold = threshold / np.max(
        overlap_count
    )  # larger than threshold in at least one snippet
    binary_prediction = (aggregated_predictions > adjusted_threshold).astype(int)
    row_starts = []
    row_stops = []
    label_names = []
    for i, label_name in enumerate(calls):
        if sum(binary_prediction[:, i]) > 0:
            row_start, row_stop = find_consecutive_ones(binary_prediction[:, i])
            row_starts += list(row_start)
            row_stops += list(row_stop)
            label_names += [label_name] * len(row_start)
    return row_starts, row_stops, label_names


def compute_labels(
    row_starts: list[int],
    row_stops: list[int],
    label_names: list[str],
    time_steps_per_output_step: int,
    label_suffix: str | None,
) -> pd.DataFrame:
    if (label_suffix is not None) & (label_suffix != ""):
        label_names = [label + label_suffix for label in label_names]
    predicted_labels = (
        pd.DataFrame(
            {
                "start": np.asarray(row_starts) * time_steps_per_output_step,
                "stop": np.asarray(row_stops) * time_steps_per_output_step,
                "label": label_names,
            }
        )
        .sort_values(by=["start", "stop", "label"])
        .reset_index(drop=True)
    )
    return predicted_labels


def _convert_times_to_seconds(
    predicted_labels: pd.DataFrame,
    delta_t: float,
) -> pd.DataFrame:
    """
    Converts the start and stop times of predicted labels from time steps to seconds.
    Parameters
    ----------
    predicted_labels : pd.DataFrame
        DataFrame with predicted labels containing 'start' and 'stop' columns in time steps.
    delta_t : float
        Time step duration in seconds.
    time_steps_per_output_step : int
        Number of time steps per output step.
    Returns
    -------
    pd.DataFrame
        DataFrame with 'start' and 'stop' columns converted to seconds.
    """
    predicted_labels.loc[:, "start"] = predicted_labels.loc[:, "start"] * delta_t
    predicted_labels.loc[:, "stop"] = predicted_labels.loc[:, "stop"] * delta_t
    return predicted_labels


def predict_wav(
    recording_path: Path | str,
    channel: int,
    model: keras.Model,
    orcai_parameter: dict,
    shape: dict,
    label_suffix: str = "*",
    msgr: Messenger = Messenger(verbosity=0),
    progressbar: tqdm = None,
) -> pd.DataFrame:
    """
    Predicts calls in a single wav file.

    Parameter
    ----------
    recording_path : Path | str
        Path to the wav file.
    channel : int
        Channel of the wav file.
    model : keras.Model
        Model for prediction.
    orcai_parameter : dict
        orcAI parameter dictionary.
    shape : dict
        Model shape dictionary.
    output_path : (Path | str) | "default"
        Path to the output file or "default" to save in the same directory as the wav file.
    overwrite : bool
        Overwrite the output file if it exists.
    save_prediction_probabilities : bool
        Save prediction probabilities to output_path
    call_duration_limits : (Path | str) | dict | None
        Path to a JSON file containing a dictionary with call duration limits for filtering. None for no filtering.
    label_suffix : str
        Suffix to add to the predicted calls.
    msgr : Messenger
        Messenger object for logging.
    progressbar : tqdm
        Progressbar object.

    Returns
    -------
    predicted_labels : pd.DataFrame
        DataFrame with predicted labels.
    aggregated_predictions : np.ndarray
        Array with aggregated predictions.
    delta_t : float
        Time step duration in seconds.

    Raises
    ------
    ValueError
        If the frequency dimensions of the spectrogram do not match the input shape.
    """

    # Generating spectrogram
    if progressbar:
        progressbar.set_description(f"{recording_path.stem}: Generating spectrogram")
        progressbar.refresh()
    spectrogram, _, times = make_spectrogram(
        recording_path, channel, orcai_parameter, msgr=msgr
    )
    delta_t = times[1] - times[0]
    if spectrogram.shape[1] != shape["input_shape"][1]:
        raise ValueError(
            f"Spectrogram shape ({spectrogram.shape[1]}) for {recording_path.stem} not equal to input shape ({shape['input_shape'][1]})"
        )

    # Prediction
    msgr.part(f"Prediction of annotations for wav_file: {recording_path.stem}")
    if progressbar:
        progressbar.set_description(f"{recording_path.stem} - Predicting annotations")
        progressbar.refresh()

    aggregated_predictions, overlap_count = compute_aggregated_predictions(
        recording_path=recording_path,
        spectrogram=spectrogram,
        model=model,
        orcai_parameter=orcai_parameter,
        shape=shape,
        msgr=msgr,
        progressbar=progressbar,
    )

    row_starts, row_stops, label_names = compute_binary_predictions(
        aggregated_predictions=aggregated_predictions,
        overlap_count=overlap_count,
        calls=orcai_parameter["calls"],
        threshold=0.5,
    )

    msgr.info("converting binary predictions into start and stop frames")
    time_steps_per_output_step = 2 ** len(orcai_parameter["model"]["filters"])
    predicted_labels = compute_labels(
        row_starts,
        row_stops,
        label_names,
        time_steps_per_output_step=time_steps_per_output_step,
        label_suffix=label_suffix,
    )
    msgr.info(f"found {len(predicted_labels)} acoustic signals")

    msgr.success("Prediction finished.")

    return predicted_labels, aggregated_predictions, delta_t


def save_predictions(
    predicted_labels: pd.DataFrame,
    output_path: Path | str,
    delta_t: float,
    msgr: Messenger = Messenger(verbosity=0),
) -> None:
    """
    Saves the predicted labels to a file.

    Parameters
    ----------
    predicted_labels : pd.DataFrame
        DataFrame with predicted labels containing 'start', 'stop', and 'label' columns.
    output_path : Path | str
        Path to the output file.
    delta_t : float
        Time step duration in seconds.
    msgr : Messenger
        Messenger object for logging.
    """
    predicted_labels = _convert_times_to_seconds(predicted_labels, delta_t)
    predicted_labels[["start", "stop", "label"]].round(4).to_csv(
        output_path, sep="\t", index=False
    )
    msgr.info(f"Predictions saved to {output_path}")
    return


def save_prediction_probabilities(
    aggregated_predictions: np.ndarray,
    orcai_parameter: dict,
    delta_t: float,
    output_path: Path | str,
    msgr: Messenger = Messenger(verbosity=0),
) -> None:
    """
    Saves the prediction probabilities to a file.
    Parameters
    ----------
    aggregated_predictions : np.ndarray
        Array with aggregated predictions.
    orcai_parameter : dict
        orcAI parameter dictionary.
    delta_t : float
        Time step duration in seconds.
    output_path : Path | str
        Path to the output file.
    msgr : Messenger
        Messenger object for logging.
    """
    predictions_path = output_path.with_name(f"{output_path.stem}_probabilities.csv.gz")
    pd.DataFrame(
        aggregated_predictions,
        columns=orcai_parameter["calls"],
        index=delta_t * range(len(aggregated_predictions)),
    ).to_csv(predictions_path, index_label="time", compression="gzip")
    msgr.info(f"Prediction probabilities saved to {predictions_path}")
    return


def _predict_and_save(
    recording_path: Path | str,
    channel: int,
    model: keras.Model,
    orcai_parameter: dict,
    shape: dict,
    output_path: Path | str = "default",
    overwrite: bool = False,
    save_probabilities: bool = False,
    call_duration_limits: (Path | str) | dict = None,
    label_suffix: str = "*",
    msgr: Messenger = Messenger(verbosity=0),
    progressbar: tqdm = None,
) -> None:
    """
    Predicts calls in a single wav file.

    Parameter
    ----------
    recording_path : Path | str
        Path to the wav file.
    channel : int
        Channel of the wav file.
    model : keras.Model
        Model for prediction.
    orcai_parameter : dict
        orcAI parameter dictionary.
    shape : dict
        Model shape dictionary.
    output_path : (Path | str) | "default"
        Path to the output file or "default" to save in the same directory as the wav file.
    overwrite : bool
        Overwrite the output file if it exists.
    save_probabilities : bool
        Save prediction probabilities to output_path
    call_duration_limits : (Path | str) | dict | None
        Path to a JSON file containing a dictionary with call duration limits for filtering. None for no filtering.
    label_suffix : str
        Suffix to add to the predicted calls.
    msgr : Messenger
        Messenger object for logging.
    progressbar : tqdm
        Progressbar object.

    Raises
    ------
    ValueError
        If the frequency dimensions of the spectrogram do not match the input shape.
    FileExistsError
        If the output_path already exists.
    """
    if output_path is not None:
        if output_path == "default":
            filename = f"{recording_path.stem}_c{channel}_{orcai_parameter['name']}_predicted.txt"
            output_path = recording_path.with_name(filename)
        else:
            output_path = Path(output_path)
        msgr.info(f"Output file: {output_path}")
        if output_path.exists():
            if overwrite:
                msgr.warning(f"Output file {output_path} already exists. Overwriting.")
            else:
                raise FileExistsError(f"Annotation file already exists: {output_path}")

    predicted_labels, aggregated_predictions, delta_t = predict_wav(
        recording_path=recording_path,
        channel=channel,
        model=model,
        orcai_parameter=orcai_parameter,
        shape=shape,
        label_suffix=label_suffix,
        msgr=msgr,
        progressbar=progressbar,
    )

    if call_duration_limits is not None:
        predicted_labels = filter_predictions(
            predicted_labels,
            delta_t=delta_t,
            call_duration_limits=call_duration_limits,
            label_suffix=label_suffix,
            msgr=msgr,
        )

    save_predictions(
        predicted_labels=predicted_labels,
        output_path=output_path,
        delta_t=delta_t,
        msgr=msgr,
    )

    if save_probabilities:
        save_prediction_probabilities(
            aggregated_predictions=aggregated_predictions,
            orcai_parameter=orcai_parameter,
            delta_t=delta_t,
            output_path=output_path,
            msgr=msgr,
        )


def predict(
    recording_path: str | Path,
    channel: int = 1,
    model_dir: str | Path = files("orcAI.models").joinpath("orcai-v1"),
    output_path: str | Path = "default",
    overwrite: bool = False,
    save_probabilities: bool = False,
    base_dir_recording: str | Path | None = None,
    call_duration_limits: str | Path | None = None,
    label_suffix: str = "*",
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> None:
    """
    Predicts calls in a wav file or a list of wav files.

    Parameter
    ----------
    recording_path : str | Path
        Path to the wav file or a CSV file of the recording table.
    channel : int
        Channel of the wav file if single wav file. If a csv is given,
        this is the default channel used.
    model_dir : str | Path
        Path to the directory containing the model.
    output_path : str | Path
        Path to the output file or "default" to save in the same directory as the wav file.
    overwrite : bool
        Overwrite the output file if it exists.
    save_probabilities : bool
        Save prediction probabilities to output_path
    base_dir_recording : str | Path | None
        Base directory for the recordings. If not given, the base directory is taken from the recording table.
    call_duration_limits : str | Path | None
        Path to a JSON file containing a dictionary with call duration limits for filtering. None for no filtering.
    label_suffix : str
        Suffix to add to the predicted calls.
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    msgr : Messenger
        Messenger object for logging. If None a new Messenger object is created.

    Returns
    -------
    None
        Saves the predicted labels to a file.

    Raises
    ------
    ValueError
        If the recording file is not a wav or csv file.
    """
    if msgr is None:
        msgr = Messenger(
            verbosity=verbosity,
            title="Predicting calls",
        )

    model_dir = Path(model_dir)
    recording_path = Path(recording_path)
    msgr.part(f"Loading model: {model_dir.stem}")

    model, orcai_parameter, shape = load_orcai_model(model_dir)

    if recording_path.suffix == ".wav":
        return _predict_and_save(
            recording_path=recording_path,
            channel=channel,
            model=model,
            orcai_parameter=orcai_parameter,
            shape=shape,
            output_path=output_path,
            overwrite=overwrite,
            save_probabilities=save_probabilities,
            call_duration_limits=call_duration_limits,
            label_suffix=label_suffix,
            msgr=msgr,
            progressbar=None,
        )
    elif recording_path.suffix == ".csv":
        recording_table = pd.read_csv(recording_path)
    else:
        raise ValueError("Recording file must be a wav or csv file")

    if base_dir_recording is not None:
        recording_table["base_dir_recording"] = base_dir_recording

    if (output_path is not None) & (output_path != "default"):
        recording_table["output_path"] = [
            Path(output_path).joinpath(
                recording + "_" + model_dir.stem + "_predicted.txt"
            )
            for recording in recording_table["recording"]
        ]
    else:
        recording_table["output_path"] = output_path

    msgr.part(f"Predicting annotations for {len(recording_table)} wav files")
    progressbar = tqdm(recording_table.index, desc="Starting ...", unit="file")
    for i in progressbar:
        try:
            _predict_and_save(
                recording_path=Path(
                    recording_table.loc[i, "base_dir_recording"]
                ).joinpath(recording_table.loc[i, "rel_recording_path"]),
                channel=recording_table.loc[i, "channel"],
                model=model,
                orcai_parameter=orcai_parameter,
                shape=shape,
                output_path=recording_table.loc[i, "output_path"],
                overwrite=overwrite,
                save_probabilities=save_probabilities,
                call_duration_limits=call_duration_limits,
                label_suffix=label_suffix,
                msgr=Messenger(verbosity=0),
                progressbar=progressbar,
            )
        except Exception as e:
            msgr.error(
                f"Error predicting {recording_table.loc[i, 'recording']}: {e.args[0]}"
            )
    msgr.success("Predictions finished.")
    return
