import time
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
    label_suffix : str
         Suffix that was added to label names during prediction.


    Returns
    -------
    out : str
        str "keep", "too long", or "too short"

    """
    label = calls["label"].replace(f"_{label_suffix}", "")

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

    if calls["duration"] < min_duration:
        out = "too short"
    elif calls["duration"] > max_duration:
        out = "too long"
    else:
        out = "keep"
    return out


def filter_predictions(
    predicted_labels: pd.DataFrame | Path | str,
    output_file: pd.DataFrame | Path | str | None = None,
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
    predicted_labels : (pd.DataFrame | Path | str)
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

    Raises
    ------
    ValueError
        If the output file already exists.

    """

    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Filtering predictions")
    msgr.part("Filtering predictions")

    if output_file is not None:
        if output_file == "default":
            if not isinstance(predicted_labels, (Path | str)):
                raise ValueError(
                    "Output file 'default' only allowed if predicted_labels is a file path"
                )
            filename = Path(predicted_labels).stem + "_filtered.txt"
            output_file = Path(predicted_labels).with_name(filename)
        else:
            output_file = Path(output_file)
        msgr.info(f"Output file: {output_file}")
        if output_file.exists():
            raise ValueError(f"Annotation file already exists: {output_file}")

    if isinstance(predicted_labels, (Path | str)):
        predicted_labels = pd.read_csv(predicted_labels, sep="\t", encoding="utf-8")

    predicted_labels["duration"] = predicted_labels["stop"] - predicted_labels["start"]

    msgr.debug("Call durations:")
    msgr.debug(predicted_labels[["label", "duration"]].groupby("label").describe())

    if isinstance(call_duration_limits, (Path | str)):
        call_duration_limits = read_json(call_duration_limits)
    msgr.debug("Call duration limits:")
    msgr.debug(call_duration_limits)

    # Filter calls based on duration
    msgr.part("Filtering calls based on duration")
    predicted_labels["duration_ok"] = predicted_labels.apply(
        lambda x: _check_duration(x, call_duration_limits, label_suffix), axis=1
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

    if output_file is not None:
        predicted_labels_duration_ok[["start", "stop", "label"]].to_csv(
            output_file, sep="\t", index=False
        )
        msgr.info(f"Filtered predictions saved to {output_file}")

    msgr.success("Filtering predictions finished.")

    return predicted_labels_duration_ok


def _predict_wav(
    recording_path: Path | str,
    channel: int,
    model: keras.Model,
    orcai_parameter: dict,
    shape: dict,
    output_path: Path | str = "default",
    save_prediction_probabilities: bool = False,
    call_duration_limits: (Path | str) | dict = None,
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
            raise FileExistsError(f"Annotation file already exists: {output_path}")

    # Generating spectrogram
    if progressbar:
        progressbar.set_description(f"{recording_path.stem}: Generating spectrogram")
        progressbar.refresh()
    spectrogram, _, times = make_spectrogram(
        recording_path, channel, orcai_parameter, msgr=msgr
    )
    if spectrogram.shape[1] != shape["input_shape"][1]:
        raise ValueError(
            f"Spectrogram shape ({spectrogram.shape[1]}) for {recording_path.stem} not equal to input shape ({shape['input_shape'][1]})"
        )

    # Prediction
    msgr.part(f"Prediction of annotations for wav_file: {recording_path.stem}")
    if progressbar:
        progressbar.set_description(f"{recording_path.stem} - Predicting annotations")
        progressbar.refresh()
    start_time = time.time()
    # Parameter
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

    # Step 5: Average the overlapping predictions (or apply another pooling function)
    msgr.info("Computing binary predictions")
    valid_mask = overlap_count > 0  # removes zeros at end
    aggregated_predictions[valid_mask] /= overlap_count[valid_mask, np.newaxis]
    threshold = 0.5 / np.max(overlap_count)  # larger than 0.5 in at least one snippet
    binary_prediction = np.zeros((total_time_steps, shape["num_labels"])).astype(int)
    binary_prediction[aggregated_predictions > threshold] = 1

    # Step 6: compute for each label in binary_prediction start and end of consecutive entries of ones
    msgr.info("converting binary predictions into start and stop times")
    delta_t = times[1] - times[0]
    row_starts = []
    row_stops = []
    label_names = []
    for i, label_name in enumerate(orcai_parameter["calls"]):
        if sum(binary_prediction[:, i]) > 0:
            row_start, row_stop = find_consecutive_ones(binary_prediction[:, i])
            row_starts += list(row_start)
            row_stops += list(row_stop)
            label_names += [label_name] * len(row_start)
    if (label_suffix is not None) & (label_suffix != ""):
        label_names = [label + label_suffix for label in label_names]
    predicted_labels = pd.DataFrame(
        {
            "start": np.asarray(row_starts) * delta_t * time_steps_per_output_step,
            "stop": np.asarray(row_stops) * delta_t * time_steps_per_output_step,
            "label": label_names,
        }
    ).sort_values(by=["start", "stop", "label"])
    msgr.info(f"found {len(predicted_labels)} acoustic signals", indent=1)
    msgr.info(
        f"time for prediction and preparing annotation file: {time.time() - start_time:.2f}"
    )

    if call_duration_limits is not None:
        predicted_labels = filter_predictions(
            predicted_labels,
            output_file=None,
            call_duration_limits=call_duration_limits,
            label_suffix=label_suffix,
            msgr=msgr,
        )

    if output_path is not None:
        predicted_labels.round(4).to_csv(output_path, sep="\t", index=False)
        msgr.success(f"Prediction finished.\nPredictions saved to {output_path}")
        msgr.success(f"Predictions saved to {output_path}")
        if save_prediction_probabilities:
            predictions_path = output_path.with_name(
                f"{output_path.stem}_probabilities.csv.gz"
            )

            delta_t * time_steps_per_output_step * range(len(aggregated_predictions))

            pd.DataFrame(
                aggregated_predictions,
                columns=orcai_parameter["calls"],
                index=delta_t
                * time_steps_per_output_step
                * range(len(aggregated_predictions)),
            ).to_csv(predictions_path, index_label="time", compression="gzip")
            msgr.success(f"Prediction probabilities saved to {predictions_path}")
    else:
        msgr.success("Prediction finished.")
    return predicted_labels


def predict(
    recording_path: str | Path,
    channel: int = 1,
    model_dir: str | Path = files("orcAI.models").joinpath("orcai-v1"),
    output_path: str | Path = "default",
    save_prediction_probabilities: bool = False,
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
    save_prediction_probabilities : bool
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
        return _predict_wav(
            recording_path=recording_path,
            channel=channel,
            model=model,
            orcai_parameter=orcai_parameter,
            shape=shape,
            output_path=output_path,
            save_prediction_probabilities=save_prediction_probabilities,
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
            _predict_wav(
                recording_path=Path(
                    recording_table.loc[i, "base_dir_recording"]
                ).joinpath(recording_table.loc[i, "rel_recording_path"]),
                channel=recording_table.loc[i, "channel"],
                model=model,
                orcai_parameter=orcai_parameter,
                shape=shape,
                output_path=recording_table.loc[i, "output_path"],
                save_prediction_probabilities=save_prediction_probabilities,
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
