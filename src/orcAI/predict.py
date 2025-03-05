import sys
import pandas as pd
import numpy as np
from pathlib import Path
from importlib.resources import files
import time
import tensorflow as tf


from orcAI.auxiliary import Messenger, read_json, find_consecutive_ones
from orcAI.architectures import (
    build_cnn_res_lstm_arch,
    masked_binary_accuracy,
    masked_binary_crossentropy,
)
from orcAI.spectrogram import make_spectrogram


def predict(
    wav_file_path,
    model_path=files("orcAI.models").joinpath("orcai-V1"),
    output_file="default",
    spectrogram_parameter=files("orcAI.defaults").joinpath(
        "default_spectrogram_parameter.json"
    ),
    channel=1,
    label_suffix="orcai-V1",
    verbosity=2,
):
    """
    Predicts annotations for a given wav file and saves them in an output file.

    Parameters
    ----------
    wav_file_path : (Path | Str)
        Path to the wav file.
    model_path : (Path | Str)
        Path to the model directory.
    output_file : (Path | Str) | "default" | None
        Path to the output file or "default" to save in the same directory as the wav file. None to not save predictions to disk.
    spectrogram_parameter : dict | (Path | Str)
        Dict containing spectrogram parameter or path to json containing the same. Defaults to default_spectrogram_parameter.json.
    channel : int
        Overwrite channel to use for prediction. If None, channel from spectrogram_parameter is used.
    label_suffix : str
        Suffix to add to the label names.
    verbosity : int
        Verbosity level.

    Returns
    -------
    df_predicted_labels : pd.DataFrame
        DataFrame with predicted labels

    """
    msgr = Messenger(verbosity=verbosity)
    msgr.part(f"Predicting annotations for wav file: {wav_file_path}")

    model_path = Path(model_path)
    msgr.info(f"Model: {model_path.stem}")
    msgr.info(f"Wav file: {wav_file_path}")

    if output_file is not None:
        if output_file == "default":
            filename = (
                Path(wav_file_path).stem + "_" + model_path.stem + "_predicted.txt"
            )
            output_file = Path(wav_file_path).with_name(filename)
        else:
            output_file = Path(output_file)
        msgr.info(f"Output file: {output_file}")
        if output_file.exists():
            msgr.error(f"Annotation file already exists: {output_file}")
            sys.exit()

    label_calls = read_json(model_path.joinpath("trained_calls.json"))
    msgr.debug("Calls for labeling:")
    msgr.debug(label_calls)

    shape = read_json(model_path.joinpath("model_shape.json"))
    msgr.debug(f"Input shape:")
    msgr.debug(shape)

    model_parameter = read_json(model_path.joinpath("model_parameter.json"))
    msgr.debug(f"Model parameter:")
    msgr.debug(model_parameter)

    if isinstance(spectrogram_parameter, (Path | str)):
        spectrogram_parameter = read_json(spectrogram_parameter)
    if channel is not None:
        spectrogram_parameter["channel"] = channel
    msgr.debug(f"Spectrogram parameters:")
    msgr.debug(spectrogram_parameter)

    msgr.part(f"Loading model: {model_path.stem}")

    msgr.info("Building model architecture")
    model = build_cnn_res_lstm_arch(**shape, **model_parameter)

    msgr.info("Loading model weights")
    model.load_weights(model_path.joinpath("model_weights.h5"))

    msgr.info("Compiling model")
    masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
        fn=lambda y_true, y_pred: masked_binary_accuracy(
            y_true, y_pred, mask_value=-1.0
        ),
        name="masked_binary_accuracy",
    )
    model.compile(
        optimizer="adam",
        loss=lambda y_true, y_pred: masked_binary_crossentropy(
            y_true, y_pred, mask_value=-1.0
        ),
        metrics=[masked_binary_accuracy_metric],
    )

    # Generating spectrogram
    spectrogram, _, times = make_spectrogram(wav_file_path, spectrogram_parameter)
    if spectrogram.shape[1] != shape["input_shape"][1]:
        msgr.error(
            f"Frequency dimensions of spectrogram shape ({spectrogram.shape[1]}) "
            + f"not equal to frequency dimension of input shape ({shape['input_shape'][1]})"
        )
        exit

    # Prediction
    msgr.part(f"Prediction of annotations for wav_file: {wav_file_path.stem}")
    start_time = time.time()
    # Parameters
    snippet_length = shape["input_shape"][0]  # Time steps in a single snippet
    shift = snippet_length // 2  # Shift time steps for overlapping windows
    time_steps_per_output_step = 2**4  # TODO: MAGIC NUMBER
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
        snippets, verbose=0 if verbosity < 2 else 1
    )  # Shape: (num_snippets, 46, 7)

    # Step 3: Initialize arrays for aggregating predictions
    msgr.info("Aggregating predictions")
    total_time_steps = spectrogram.shape[0] // time_steps_per_output_step
    aggregated_predictions = np.zeros(
        (total_time_steps, shape["num_labels"])
    )  # Shape: (3600 * 46, 7)
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
    valid_mask = overlap_count > 0
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
    for i, label_name in enumerate(label_calls):
        if sum(binary_prediction[:, i]) > 0:
            row_start, row_stop = find_consecutive_ones(binary_prediction[:, i])
            row_starts += list(row_start)
            row_stops += list(row_stop)
            label_names += [label_name] * len(row_start)
    if (label_suffix is not None) & (label_suffix != ""):
        label_names = [label + "_" + label_suffix for label in label_names]
    predicted_labels = pd.DataFrame(
        {
            "label": label_names,
            "start": np.asarray(row_starts) * delta_t * time_steps_per_output_step,
            "stop": np.asarray(row_stops) * delta_t * time_steps_per_output_step,
        }
    )
    msgr.info(f"found {len(predicted_labels)} acoustic signals", indent=1)
    msgr.info(
        f"time for prediction and preparing annotation file: {time.time()-start_time:.2f}"
    )
    if output_file is not None:
        predicted_labels[["start", "stop", "label"]].to_csv(
            output_file, sep="\t", index=False
        )
        msgr.success(f"Prediction finished.\nPredictions saved to {output_file}")
    else:
        msgr.success(f"Prediction finished.")
    return predicted_labels


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
    # print(x)
    # print(call_duration_limits)
    label = x["label"].replace(f"_{label_suffix}", "")
    # print(label)
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
        Verbosity level.

    Returns
    -------
    predicted_labels_duration_ok : pd.DataFrame
        DataFrame with predicted labels after filtering based
        on duration.
    """

    msgr = Messenger(verbosity=verbosity)
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
