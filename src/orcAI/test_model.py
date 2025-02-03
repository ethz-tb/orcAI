# %%
#  import
import os
import zarr
import sys
import random
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import auxiliary as aux
import model as mod
import plot


computer = "laptop"
train_model = False
scratch_dir = "undefined"
mode_dataset = "use_existing"
interactive = True

interactive = aux.check_interactive()
if not interactive:
    print("not interactive")
    (computer, data_dir, model_name, project_dir) = aux.train_model_commandline_parse()
else:
    project_dir = "/Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/"
    computer = "laptop"
    data_dir = "/Users/sb/AI_data/tvtdata/"
    model_name = "cnn_res_lstm_model"

# %%
# Read parameters
print("Project directory:", project_dir)
os.chdir(project_dir)
print("READ IN PARAMETERS")
dicts = {
    "directories_dict": "GenericParameters/directories.dict",
    "model_dict": project_dir + "Results/" + model_name + "/model.dict",
    "calls_for_labeling_list": "GenericParameters/calls_for_labeling.list",
}
for key, value in dicts.items():
    print("  - reading", key)
    globals()[key] = aux.read_dict(value, True)

# %%
# read in test.csv.gz and add t_start and t_stop
test_df = pd.read_csv(directories_dict[computer]["root_dir_tvtdata"] + "test.csv.gz")
fn_times = test_df["fnstem_path"].iloc[0] + "spectrogram/times.json"
t_vector = aux.read_json_to_vector(fn_times)
delta_t = t_vector[1] - t_vector[0]
test_df["t_start"] = test_df["row_start"] * delta_t
test_df["t_stop"] = test_df["row_stop"] * delta_t
test_df["t_start"] = test_df["t_start"].apply(aux.seconds_to_hms)
test_df["t_stop"] = test_df["t_stop"].apply(aux.seconds_to_hms)
test_df


# %%
# function to get spec and reshaped labels from index of test_df
def get_spec_labels(df, index, predict=False):
    from pathlib import Path

    spec_path = os.path.join(df["fnstem_path"].iloc[index], "spectrogram/zarr.spc")
    lbl_path = os.path.join(df["fnstem_path"].iloc[index], "labels/zarr.lbl")
    spectrogram_z = zarr.open(spec_path, mode="r")
    labels_z = zarr.open(lbl_path, mode="r")
    spec = spectrogram_z[df["row_start"].iloc[index] : df["row_stop"].iloc[index], :]
    labels_long = labels_z[df["row_start"].iloc[index] : df["row_stop"].iloc[index], :]
    lab_true = mod.reshape_labels(labels_long, len(model_dict["filters"]))
    sp = spec.reshape((1, spec.shape[0], spec.shape[1], 1))
    if predict:
        lab_pred = model.predict(sp, verbose=0)
        lab_pred = lab_pred[0, :, :]
        lab_pred = (lab_pred + 0.5).astype(int)
    title = (
        Path(df["fnstem_path"].iloc[index]).stem
        + "    "
        + df["t_start"].iloc[index]
        + "-"
        + df["t_stop"].iloc[index]
    )
    if predict:
        return spec, lab_true, lab_pred, title
    else:
        return spec, lab_true, None, title


predict = False
spec, lab_true, lab_pred, title = get_spec_labels(test_df, 0, predict)

# %%
# Build  model
print("Building model:")

model_choice_dict = {
    "cnn_res_model": lambda: mod.build_cnn_res_model(
        input_shape,
        num_labels,
        model_dict["filters"],
        model_dict["kernel_size"],
        model_dict["dropout_rate"],
    ),
    "cnn_res_lstm_model": lambda: mod.build_cnn_res_lstm_model(
        input_shape,
        num_labels,
        model_dict["filters"],
        model_dict["kernel_size"],
        model_dict["dropout_rate"],
        model_dict["lstm_units"],
    ),
    "cnn_res_transformer_model": lambda: mod.build_cnn_res_transformer_model(
        input_shape,
        num_labels,
        model_dict["filters"],
        model_dict["kernel_size"],
        model_dict["dropout_rate"],
        model_dict["num_heads"],
    ),
}
input_shape = (spec.shape[0], spec.shape[1], 1)  #  shape
num_labels = lab_true.shape[1]  # Number of sound types

model = mod.build_model(model_choice_dict, model_dict, input_shape, num_labels)


# %%
# Loading model weights
print("Loading weights from stored model:", model_dict["name"])
model.load_weights(
    project_dir + "Results/" + model_dict["name"] + "/" + model_dict["name"]
)


# %%
# Compiling Model
print("Compiling model:", model_dict["name"])
# Metric
masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
    fn=lambda y_true, y_pred: mod.masked_binary_accuracy(
        y_true, y_pred, mask_value=-1.0
    ),
    name="masked_binary_accuracy",
)
# Callbacks
early_stopping = EarlyStopping(
    monitor="val_masked_binary_accuracy",  # Use the validation metric
    patience=model_dict["patience"],  # Number of epochs to wait for improvement
    mode="max",  # Stop when accuracy stops increasing
    restore_best_weights=True,  # Restore weights from the best epoch
)
model_checkpoint = ModelCheckpoint(
    model_dict["name"],
    monitor="val_masked_binary_accuracy",
    save_best_only=True,
    save_weights_only=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_masked_binary_accuracy",  # Monitor your custom validation metric
    factor=0.5,  # Reduce learning rate by a factor of 0.5
    patience=3,  # Wait for 3 epochs of no improvement
    min_lr=1e-6,  # Set a lower limit for the learning rate
    verbose=1,  # Print updates to the console
)

model.compile(
    optimizer="adam",
    loss=lambda y_true, y_pred: mod.masked_binary_crossentropy(
        y_true, y_pred, mask_value=-1.0
    ),
    metrics=[masked_binary_accuracy_metric],
)
aux.print_memory_usage()

# %%
# show spec, lab_true, lab_pred for random element of test_df
random_index = random.randint(0, len(test_df))
predict = True
spec1, lab_true, lab_pred, title = get_spec_labels(test_df, random_index, predict)

# lower_quantile, upper_quantile = np.quantile(spec, spectrogram_dict['quantiles'])
# clipped_spec = np.clip(spec, lower_quantile, upper_quantile)
# max_val = np.max(clipped_spec)
# min_val = np.min(clipped_spec)
# clipped_normed_spec = (clipped_spec-min_val)/(max_val - min_val)
plot.plot_spec_and_labels(
    spec,
    calls_for_labeling_list,
    lab_true,
    lab_pred,
    title,
)

# %%
extracted_snippets = pd.read_csv("extracted_snippets.csv.gz")
from matplotlib import pyplot as plt

fns = extracted_snippets["fnstem"].iloc[1]
plt.plot(
    extracted_snippets["row_start"][
        (extracted_snippets["fnstem"] == fns) & (extracted_snippets["type"] == "train")
    ],
    extracted_snippets["row_stop"][
        (extracted_snippets["fnstem"] == fns) & (extracted_snippets["type"] == "train")
    ],
    marker=".",
    linestyle="",
    color="red",
)
plt.plot(
    extracted_snippets["row_start"][
        (extracted_snippets["fnstem"] == fns) & (extracted_snippets["type"] == "test")
    ],
    extracted_snippets["row_stop"][
        (extracted_snippets["fnstem"] == fns) & (extracted_snippets["type"] == "test")
    ],
    marker=".",
    linestyle="",
    color="green",
)
plt.plot(
    extracted_snippets["row_start"][
        (extracted_snippets["fnstem"] == fns) & (extracted_snippets["type"] == "val")
    ],
    extracted_snippets["row_stop"][
        (extracted_snippets["fnstem"] == fns) & (extracted_snippets["type"] == "val")
    ],
    marker=".",
    linestyle="",
    color="blue",
)

# %%
# Parameters
snippet_length = spec.shape[0]  # Time steps in a single snippet
shift = snippet_length // 2  # Shift time steps for overlapping windows
num_labels = lab_pred.shape[1]  # Number of label types
prediction_length = lab_pred.shape[0]  # Output time steps per prediction
time_steps_per_output_step = snippet_length // prediction_length

# wavfile = "/Volumes/OrcAI-Disk/Acoustics/2023_dtag/oo23_184a008.wav"

# full_spectrogram, frequencies, times = spec.create_spectrogram(wavfile, spectrogram_dict)

zarr_file = zarr.open(
    "/Users/sb/AI_data/spectrograms/wo_annot/oo23_184a008/spectrogram/zarr.spc",
    mode="r",
)
full_spectrogram = zarr_file[:]
# Step 1: Create overlapping spectrogram snippets
num_snippets = (full_spectrogram.shape[0] - snippet_length) // shift + 1
snippets = np.array(
    [
        full_spectrogram[i * shift : i * shift + snippet_length]
        for i in range(num_snippets)
    ]
)  # Shape: (num_snippets, 736, 171)

# Step 2: Model predictions for all snippets
# Reshape snippets for model input (e.g., add channel dimension if required)
snippets = snippets[..., np.newaxis]  # Shape: (num_snippets, 736, 171, 1)
predictions = model.predict(snippets)  # Shape: (num_snippets, 46, 7)

# Step 3: Initialize arrays for aggregating predictions
total_time_steps = full_spectrogram.shape[0] // time_steps_per_output_step
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
valid_mask = overlap_count > 0
aggregated_predictions[valid_mask] /= overlap_count[valid_mask, np.newaxis]
threshold = 0.5 / np.max(overlap_count)  # larger than 0.5 in at least one snippet
binary_prediction = np.zeros((total_time_steps, num_labels)).astype(int)
binary_prediction[aggregated_predictions > threshold] = 1

# Step 6: compute for each label in binary_prediction start and end of consecutive entries of ones
calls_for_labeling_list

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
df_predicted_labels
print(len(df_predicted_labels))
df_predicted_labels[(df_predicted_labels["stop"] - df_predicted_labels["start"]) > 0.1]
tmp = df_predicted_labels.sort_values(by="start").reset_index().drop(columns=["index"])
tmp[["start", "stop", "label"]].to_csv("tmp.txt", sep="\t", index=False)


# %%
import json
import os
import pandas as pd

# Path to the hyperparameter logs directory
log_dir = "cnn_res_lstm_model/hyperparameter_logs/"

# Find all trial files
trial_files = [f for f in os.listdir(log_dir) if f.startswith("trial_")]

# Initialize a list to store trial data
trial_data = []

# Loop through each trial file
for trial_file in trial_files:
    trial_path = os.path.join(log_dir, trial_file)

    # Load the trial data
    with open(trial_path + "/trial.json", "r") as f:
        trial_info = json.load(f)

    # Extract hyperparameters
    hps = trial_info["hyperparameters"]["values"]  # Hyperparameters for this trial

    # Extract metrics
    metrics = trial_info.get("metrics", {}).get("metrics", {})
    if "val_masked_binary_accuracy" in metrics:
        val_accuracy = metrics["val_masked_binary_accuracy"]["observations"][0]["value"]
    else:
        val_accuracy = None

    # Append trial data
    trial_data.append({"trial_id": trial_file, **hps, "val_accuracy": val_accuracy})

# Convert to a DataFrame for analysis
df = pd.DataFrame(trial_data)

# Display the DataFrame
print(df)

# Sort by validation accuracy
df_sorted = df.sort_values(by="val_accuracy", ascending=False)
print("Top Trials by Validation Accuracy:")
print(df_sorted)


# %%
pred_tmp = model.predict(snippets[21:22, :, :, :])
pred_tmp[pred_tmp > 0.5] = 1
pred_tmp[pred_tmp <= 0.5] = 0
pred_tmp = pred_tmp.astype(int)
pred_tmp
# %%
tmp = predictions[21, :, :]
tmp[tmp > 0.5] = 1
tmp[tmp <= 0.5] = 0
tmp = tmp.astype(int)
tmp
# %%
