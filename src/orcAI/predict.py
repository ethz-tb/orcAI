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
    model_path=files("orcAI.models").joinpath("orcai_Orca_1_0_0"),
    output_file=None,
    spectrogram_parameters_path=files("orcAI.defaults").joinpath(
        "default_spectrogram_parameter.json"
    ),
    verbosity=2,
):
    """
    Predicts annotations for a given wav file and saves them in an output file.

    Parameters
    ----------
    wav_file_path : Path
        Path to the wav file.
    output_file : Path
        Path to the output file or None if the output file should be saved in the same directory as the wav file.
    spectrogram_parameters_path : Path
        Path to the spectrogram parameter file.
    verbosity : int
        Verbosity level.
    """
    msgr = Messenger(verbosity=verbosity)
    msgr.part(f"Predicting annotations for wav file: {wav_file_path}")
    if output_file is None:
        output_file = Path(wav_file_path).with_suffix(".txt")
    if output_file.exists():
        msgr.error(f"Annotation file already exists: {output_file}")
        sys.exit()

    msgr.info(f"Model: {model_path.stem}")
    msgr.info(f"Output file: {output_file}")

    label_calls = read_json(model_path.joinpath("trained_calls.json"))
    msgr.info("Calls for labeling:")
    msgr.info(label_calls)

    shape = read_json(model_path.joinpath("shape.json"))
    msgr.info(f"Input shape:")
    msgr.info(shape)

    model_parameter = read_json(model_path.joinpath("model_parameter.json"))
    msgr.info(f"Model parameter:")
    msgr.info(model_parameter)

    spectrogram_parameters = read_json(spectrogram_parameters_path)
    msgr.info(f"Spectrogram parameters:")
    msgr.info(spectrogram_parameters)

    msgr.part(f"Loading model: {model_path.stem}")

    msgr.info("Building model architecture")
    model = build_cnn_res_lstm_arch(**shape, **model_parameter)

    msgr.info("Loading model weights")
    model.load_weights(model_path)

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
    spectrogram, frequencies, times = make_spectrogram(
        wav_file_path, spectrogram_parameters
    )
    if spectrogram.shape[1] != shape["input_shape"][1]:
        msgr.error(
            f"Frequency dimensions of spectrogram shape ({spectrogram.shape[1]}) "
            + f"not equal to frequency dimension of input shape ({shape['input_shape'][1]})"
        )
        exit

    # Prediction
    msgr.part("Prediction of annotations for wav_file:", wav_file_path)
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
    predictions = model.predict(snippets)  # Shape: (num_snippets, 46, 7)

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
    df_predicted_labels = pd.DataFrame(
        {
            "label": label_names,
            "start": np.asarray(row_starts) * delta_t * time_steps_per_output_step,
            "stop": np.asarray(row_stops) * delta_t * time_steps_per_output_step,
        }
    )
    msgr.info(f"found {len(df_predicted_labels)} acoustic signals", indent=1)
    msgr.info(
        f"time for prediction and preparing annotation file: {time.time()-start_time:.2f}"
    )

    minimal_duration = 0.1
    tmp = df_predicted_labels[
        (df_predicted_labels["stop"] - df_predicted_labels["start"]) < minimal_duration
    ]
    tmp = tmp.groupby(["label"]).count()
    eliminated_calls = pd.DataFrame(
        {"label": list(tmp.index), "number": list(tmp["start"])}
    )
    msgr.info(
        f"number and type signals eliminated because shorter than {minimal_duration} seconds"
    )
    msgr.info(eliminated_calls.to_string(index=False))
    df_predicted_labels = df_predicted_labels[
        (df_predicted_labels["stop"] - df_predicted_labels["start"]) >= minimal_duration
    ]
    df_predicted_labels = (
        df_predicted_labels.sort_values(by="start")
        .reset_index()
        .drop(columns=["index"])
    )
    msgr.info(f"{len(df_predicted_labels)} acoustic signals remaining")

    df_predicted_labels[["start", "stop", "label"]].to_csv(
        output_file, sep="\t", index=False
    )

    msgr.success(f"OrcAI - prediction finished. Predictions saved to {output_file}")
