#!/usr/bin/env python
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
import load


computer = "laptop"
train_model = False
scratch_dir = "undefined"
mode_dataset = "use_existing"
interactive = True

interactive = aux.check_interactive()
if not interactive:
    print("Command-line call:", " ".join(sys.argv))
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
    "model_dict": "/Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/Results/Euler/Final/cnn_res_lstm_model_final/model.dict",
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


# %%
# read in extracted snippets
extracted_snippets = pd.read_csv(project_dir + "Results/" + "extracted_snippets.csv.gz")
fnstems = list(extracted_snippets["fnstem"])
extracted_snippets["fnstem_path"] = [
    directories_dict[computer]["root_dir_spectrograms"] + x + "/" for x in fnstems
]


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
    "/Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/Results/Euler/Final/cnn_res_lstm_model_final/cnn_res_lstm_model"
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

model.compile(
    optimizer="adam",
    loss=lambda y_true, y_pred: mod.masked_binary_crossentropy(
        y_true, y_pred, mask_value=-1.0
    ),
    metrics=[masked_binary_accuracy_metric],
)

aux.print_memory_usage()

# %%
# run model on test data
test_dataset = load.reload_dataset(data_dir + "test_dataset", model_dict["batch_size"])
print("Evaluate model on test data:", model_dict["name"])
test_loss, test_metric = model.evaluate(test_dataset)
print(f"  - test loss: {test_loss}")
print(f"  - test masked binary accuracy: {test_metric}")

for spectrogram_batch, label_batch in test_dataset.take(1):
    print(f"  - spectrogram batch shape: {spectrogram_batch.shape}")
    print(f"  - Label batch shape: {label_batch.shape}")

# Confusion matrices
print(f"  - confusion matrices on test data:")
# Extract true labels
y_pred_batch = []
y_true_batch = []
i = 1
len_test_data = len(test_dataset)
print("  - predicting test data:")
for spectrogram_batch, label_batch in test_dataset:
    # print("  -", i, "of", len_test_data)
    y_true_batch.append(label_batch.numpy())
    y_pred_batch.append(model.predict(spectrogram_batch, verbose=0))
    i += 1

y_true_batch = np.concatenate(y_true_batch, axis=0)
y_pred_batch = np.concatenate(y_pred_batch, axis=0)
confusion_matrices = aux.compute_confusion_matrix(
    y_true_batch, y_pred_batch, calls_for_labeling_list, mask_value=-1
)
aux.print_confusion_matrices(confusion_matrices)
test_accuracy = mod.masked_binary_accuracy(
    y_true_batch, y_pred_batch, mask_value=-1.0
).numpy()

print(
    " - masked binary test accuracy based on select data (equivalent to train and val):",
    test_accuracy,
)

# aux.write_dict(
#     confusion_matrices,
#     project_dir + "Results/" + model_dict["name"] + "/" + "/confusion_matrices",
# )


# %%
# run model on test part of extracted_snippets
test_all_df = extracted_snippets[["fnstem_path", "row_start", "row_stop"]][
    extracted_snippets["type"] == "test"
]
test_all_df = test_all_df.sample(n=10000, replace=False).reset_index()
batch_size = model_dict["batch_size"]
n_filters = len(model_dict["filters"])
shuffle = False
test_all_loader = load.ChunkedMultiZarrDataLoader(
    test_all_df,
    batch_size=batch_size,
    n_filters=n_filters,
    shuffle=shuffle,
)
test_all_dataset = tf.data.Dataset.from_generator(
    lambda: load.data_generator(test_all_loader),
    output_signature=(
        tf.TensorSpec(
            shape=(spectrogram_batch.shape[1], spectrogram_batch.shape[2], 1),
            dtype=tf.float32,
        ),  # Single spectrogram shape
        tf.TensorSpec(
            shape=(label_batch.shape[1], label_batch.shape[2]), dtype=tf.float32
        ),  # Single label shape
    ),
)
test_all_dataset = test_all_dataset.batch(
    batch_size, drop_remainder=True
).prefetch(  # Batch size as defined in model_dict
    buffer_size=tf.data.AUTOTUNE
)
total_batches = len(test_all_loader)  # Assuming test_all_loader supports len()

test_all_dataset = test_all_dataset.apply(
    tf.data.experimental.assert_cardinality(total_batches)
)

# %%
print("Evaluate model on all snippets in test segments of wav:", model_dict["name"])
test_loss, test_metric = model.evaluate(test_all_dataset)
print(f"  - test loss: {test_loss}")
print(f"  - test masked binary accuracy: {test_metric}")

print(f"  - confusion matrices on all test snippets (not removing empty ones):")
# Extract true labels
y_pred_batch = []
y_true_batch = []
i = 1
len_test_data_all = len(test_all_dataset)
print("  - predicting test data on all snippets:")
for spectrogram_batch, label_batch in test_all_dataset:
    print("  -", i, "of", len_test_data)
    y_true_batch.append(label_batch.numpy())
    y_pred_batch.append(model.predict(spectrogram_batch, verbose=0))
    i += 1

y_true_batch = np.concatenate(y_true_batch, axis=0)
y_pred_batch = np.concatenate(y_pred_batch, axis=0)
confusion_matrices_test_all = aux.compute_confusion_matrix(
    y_true_batch, y_pred_batch, calls_for_labeling_list, mask_value=-1
)
aux.print_confusion_matrices(confusion_matrices_test_all)
test_all_accuracy = mod.masked_binary_accuracy(
    y_true_batch, y_pred_batch, mask_value=-1.0
).numpy()
print(" - masked binary test accuracy based on all data:", test_all_accuracy)

# aux.write_dict(
#     confusion_matrices,
#     project_dir + "Results/" + model_dict["name"] + "/" + "/confusion_matrices",
# )


# %%$
# generate latex table for confusion matrices
df = pd.DataFrame.from_dict(confusion_matrices, orient="index").reset_index()
df.insert(1, "set", "select")
df.rename(columns={"index": "label"}, inplace=True)
df_all = pd.DataFrame.from_dict(
    confusion_matrices_test_all, orient="index"
).reset_index()
df_all.insert(1, "set", "all")
df_all.rename(columns={"index": "label"}, inplace=True)

confusion_matrix_table = pd.concat([df, df_all])
# confusion_matrix_table = confusion_matrix_table.sort_values(['label'])
for col in ["TP", "FN", "FP", "TN"]:
    confusion_matrix_table[col] = (confusion_matrix_table[col] * 100).round(3)

print(confusion_matrix_table.to_latex(index=False))

# %%


# %%
# show spec, lab_true, lab_pred for random element of test_df
random_index = random.randint(0, len(test_df))
predict = True
spec, lab_true, lab_pred, title = get_spec_labels(test_df, random_index, predict)

lower_quantile, upper_quantile = np.quantile(spec, [0.001, 0.999])
clipped_spec = np.clip(spec, lower_quantile, upper_quantile)
max_val = np.max(clipped_spec)
min_val = np.min(clipped_spec)
clipped_normed_spec = (clipped_spec - min_val) / (max_val - min_val)
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
