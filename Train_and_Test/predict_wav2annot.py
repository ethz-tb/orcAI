#!/usr/bin/env python
# %%
#  import
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf


# import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import auxiliary as aux
import model as mod
import spectrogram as spec


interactive = True

interactive = aux.check_interactive()
if not interactive:
    print("Command-line call:", " ".join(sys.argv))
    (wav_file, pred_annot_file) = aux.predict_wav2annot_commandline_parse()
else:
    wav_file = "/Volumes/OrcAI-Disk/Acoustics/2023_dtag/oo23_184a008.wav"
    pred_annot_file = "oo23_184a008_pred.txt"
    os.chdir("/Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/")


model_path = "FinalModel/model"
# %%
# Parameters
calls_for_labeling_list = ["BR", "BUZZ", "HERDING", "PHS", "SS", "TAILSLAP", "WHISTLE"]
input_shape = (736, 171, 1)  #  shape
num_labels = 7  # Number of sound types
filters = [30, 40, 50, 60]
kernel_size = 5
dropout_rate = 0.4
lstm_units = 64


# %%
# Build  model
print("Building and compiling model:", model_path)
print("  - building model")

model = mod.build_cnn_res_lstm_model(
    input_shape, num_labels, filters, kernel_size, dropout_rate, lstm_units
)


# %%
# Loading model weights
print("  - loading weights")
model.load_weights(model_path)
# %%
# Compile Model
print("  - compiling model")
# Metric
masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
    fn=lambda y_true, y_pred: mod.masked_binary_accuracy(
        y_true, y_pred, mask_value=-1.0
    ),
    name="masked_binary_accuracy",
)
model.compile(
    optimizer="adam",
    loss=lambda y_true, y_pred: mod.masked_binary_crossentropy(
        y_true, y_pred, mask_value=-1.0
    ),
    metrics=[masked_binary_accuracy_metric],
)


# %%
# Generating spectrogram
print("Load wav_file and generate spectrogram")
print("  - spectrogram parameters:")
print("     - (down)sampling rate:", spectrogram_dict["sampling_rate"])
print("     - nfft:", spectrogram_dict["nfft"])
print("     - n_overlap:", spectrogram_dict["n_overlap"])
print("     - freq_range (Hz):", spectrogram_dict["freq_range"])
print("     - spectrogram power quantiles:", spectrogram_dict["quantiles"])
print("     - channel:", spectrogram_dict["channel"])
spectrogram_dict = {
    "sampling_rate": 48000,
    "nfft": 512,
    "n_overlap": 256,
    "freq_range": [0, 16000],
    "quantiles": [0.01, 0.999],
    "duration": 4,
    "channel": 1,
}
print(f"  - NOTE: taking channel {spectrogram_dict['channel']} for spectrogram")
spectrogram, frequencies, times = spec.create_spectrogram(wav_file, spectrogram_dict)


if spectrogram.shape[1] != input_shape[1]:
    print(
        f"WARNING: frequency dimensions of spectrogram shape ({spectrogram.shape[1]})not equal to frequency dimension \nof input shape ({input_shape[1]})"
    )
    exit


# %%
# Prediction
print("Prediction of annotations for wav_file:", wav_file)

# Parameters
snippet_length = 736  # Time steps in a single snippet
shift = snippet_length // 2  # Shift time steps for overlapping windows
time_steps_per_output_step = 2**4
prediction_length = (
    snippet_length // time_steps_per_output_step
)  # Output time steps per prediction

# Step 1: Create overlapping spectrogram snippets
num_snippets = (spectrogram.shape[0] - snippet_length) // shift + 1
print(f"  - slicing into {num_snippets} snippets for prediction")

snippets = np.array(
    [spectrogram[i * shift : i * shift + snippet_length] for i in range(num_snippets)]
)  # Shape: (num_snippets, 736, 171)

# Step 2: Model predictions for all snippets
print("  - prediction of snippets")

snippets = snippets[..., np.newaxis]  # Shape: (num_snippets, 736, 171, 1)
predictions = model.predict(snippets)  # Shape: (num_snippets, 46, 7)

# Step 3: Initialize arrays for aggregating predictions
print("  - aggregating predictions")
total_time_steps = spectrogram.shape[0] // time_steps_per_output_step
aggregated_predictions = np.zeros(
    (total_time_steps, num_labels)
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
print("  - computing binary predictions")
valid_mask = overlap_count > 0
aggregated_predictions[valid_mask] /= overlap_count[valid_mask, np.newaxis]
threshold = 0.5 / np.max(overlap_count)  # larger than 0.5 in at least one snippet
binary_prediction = np.zeros((total_time_steps, num_labels)).astype(int)
binary_prediction[aggregated_predictions > threshold] = 1

# Step 6: compute for each label in binary_prediction start and end of consecutive entries of ones
print("  - converting binary predictions into start and stop times")
delta_t = times[1] - times[0]
row_starts = []
row_stops = []
label_names = []
for i, label_name in enumerate(calls_for_labeling_list):
    if sum(binary_prediction[:, i]) > 0:
        row_start, row_stop = aux.find_consecutive_ones(binary_prediction[:, i])
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
print(f"  - found {len(df_predicted_labels)} acoustic signals")

# %%

minimal_duration = 0.1
print(f"  - eliminating calls with less then {minimal_duration} seconds duration")
tmp = df_predicted_labels[
    (df_predicted_labels["stop"] - df_predicted_labels["start"]) < minimal_duration
]
tmp = tmp.groupby(["label"]).count()
eliminated_calls = pd.DataFrame(
    {"label": list(tmp.index), "number": list(tmp["start"])}
)
print("  - eliminated signals")
print(eliminated_calls.to_string())
df_predicted_labels = df_predicted_labels[
    (df_predicted_labels["stop"] - df_predicted_labels["start"]) >= minimal_duration
]
df_predicted_labels = (
    df_predicted_labels.sort_values(by="start").reset_index().drop(columns=["index"])
)
print(f"  - {len(df_predicted_labels)} acoustic signals remaining")
fnstem = Path(wav_file).stem
print(
    f"  - saving these predicted accoustic signals as annotation file in:",
    fnstem + "_pred.txt",
)

df_predicted_labels[["start", "stop", "label"]].to_csv(
    "FinalModel/" + pred_annot_file, sep="\t", index=False
)


# %%
