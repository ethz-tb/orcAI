import sys
import pandas as pd
import numpy as np
from pathlib import Path
from importlib.resources import files
import time
import keras
from tqdm import tqdm
from keras.saving import load_model

from orcAI.auxiliary import Messenger, find_consecutive_ones
from orcAI.architectures import (
    res_net_LSTM_arch,
    masked_binary_accuracy,
    masked_binary_crossentropy,
)
from orcAI.spectrogram import make_spectrogram
from orcAI.io import read_json


def _check_duration(x, call_duration_limits, label_suffix="orcai-V1"):
    """
    Filter calls based on duration.

    Parameters
    ----------
    x : pd.DataFrame
        DataFrame with calls of a single label.
    call_duration_limits : dict
        Dictionary with call duration limits for each label.
    label_suffix : str
        Suffix to add to the label names.


    Returns
    -------
    x : str
        str "keep", "too long", or "too short"

    """
    label = x["label"].replace(f"_{label_suffix}", "")

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

    if x["duration"] < min_duration:
        out = "too short"
    elif x["duration"] > max_duration:
        out = "too long"
    else:
        out = "keep"
    return out


def filter_predictions(
    predicted_labels,
    output_file=None,
    call_duration_limits=files("orcAI.defaults").joinpath(
        "default_call_duration_limits.json"
    ),
    label_suffix="orcai-V1",
    verbosity=2,
    msgr=None,
):
    """
    Filter predictions based on duration.

    Parameters
    ----------
    predicted_labels : (pd.DataFrame | Path | Str)
        DataFrame with predicted labels or path to a file with predicted labels.
    output_file : (Path | Str) | "default" | None
        Path to the output file or "default" to save in the same directory as the predicted labels file. None to not save predictions to disk.
    call_duration_limits : (Path | Str) | dict
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

    if output_file is not None:
        if output_file == "default":
            if not isinstance(predicted_labels, (Path | str)):
                msgr.error(
                    "Output file 'default' only allowed if predicted_labels is a file path"
                )
                sys.exit()
            filename = Path(predicted_labels).stem + "_filtered.txt"
            output_file = Path(predicted_labels).with_name(filename)
        else:
            output_file = Path(output_file)
        msgr.info(f"Output file: {output_file}")
        if output_file.exists():
            msgr.error(f"Annotation file already exists: {output_file}")
            sys.exit()

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
        msgr.success(
            f"Filtering predictions finished.\nFiltered predictions saved to {output_file}"
        )
    else:
        msgr.success(f"Filtering predictions finished.")
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
):
    if output_path is not None:
        if output_path == "default":
            filename = f"{recording_path.stem}_c{channel}_{orcai_parameter['name']}_predicted.txt"
            output_path = recording_path.with_name(filename)
        else:
            output_path = Path(output_path)
        msgr.info(f"Output file: {output_path}")
        if output_path.exists():
            msgr.error(f"Annotation file already exists: {output_path}")
            sys.exit()  # TODO: replace with return

    # Generating spectrogram
    if progressbar:
        progressbar.set_description(f"{recording_path.stem}: Generating spectrogram")
        progressbar.refresh()
    spectrogram, _, times = make_spectrogram(
        recording_path, channel, orcai_parameter, msgr=msgr
    )
    if spectrogram.shape[1] != shape["input_shape"][1]:
        msgr.error(
            f"Frequency dimensions of spectrogram shape ({spectrogram.shape[1]}) "
            + f"not equal to frequency dimension of input shape ({shape['input_shape'][1]})"
        )
        sys.exit()  # TODO: replace with return

    # Prediction
    msgr.part(f"Prediction of annotations for wav_file: {recording_path.stem}")
    if progressbar:
        progressbar.set_description(f"{recording_path.stem} - Predicting annotations")
        progressbar.refresh()
    start_time = time.time()
    # Parameters
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
        f"time for prediction and preparing annotation file: {time.time()-start_time:.2f}"
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
        msgr.success(f"Prediction finished.")
    return predicted_labels


def predict(
    recording_path,
    channel=1,
    model_dir=files("orcAI.models").joinpath("orcai-v1"),
    output_path="default",
    save_prediction_probabilities=False,
    base_dir_recording=None,
    call_duration_limits=None,
    label_suffix="*",
    verbosity=2,
    msgr=None,
):
    if msgr is None:
        msgr = Messenger(
            verbosity=verbosity,
            title="Predicting calls",
        )

    model_dir = Path(model_dir)
    recording_path = Path(recording_path)
    msgr.part(f"Loading model: {model_dir.stem}")

    orcai_parameter = read_json(model_dir.joinpath("orcai_parameter.json"))
    shape = read_json(model_dir.joinpath("model_shape.json"))

    if model_dir.joinpath(orcai_parameter["name"] + ".keras").exists():
        model = load_model(
            model_dir.joinpath(orcai_parameter["name"] + ".keras"),
            custom_objects=None,
            compile=True,
            safe_mode=True,
        )
    elif model_dir.joinpath("model_weights.h5").exists():
        # legacy model
        model = res_net_LSTM_arch(**shape, **orcai_parameter["model"])
        model.load_weights(model_dir.joinpath("model_weights.h5"))
        masked_binary_accuracy_metric = keras.metrics.MeanMetricWrapper(
            fn=masked_binary_accuracy,
            name="masked_binary_accuracy",
        )
        model.compile(
            optimizer="adam",
            loss=masked_binary_crossentropy,
            metrics=[masked_binary_accuracy_metric],
        )
    else:
        msgr.error("Couldn't find model weights or keras model file in {model_dir}")
        sys.exit()

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
        msgr.error("Recording file must be a wav or csv file")
        sys.exit()

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
        _predict_wav(
            recording_path=Path(recording_table.loc[i, "base_dir_recording"]).joinpath(
                recording_table.loc[i, "rel_recording_path"]
            ),
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
    msgr.success("Predictions finished.")
    return
