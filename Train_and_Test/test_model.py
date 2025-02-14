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
# Functions to compute misclassification matrix
def stack_batch(batch):
    stacked = np.vstack(batch).astype(int)
    return stacked


def get_mask_for_rows_with_atmost_one_1(mat):
    count_ones_per_row = np.sum(mat == 1, axis=1)
    # Create a boolean mask: True if the row has <= 1 '1'
    mask = count_ones_per_row <= 1
    return mask


def compute_misclassification_table(mat1, mat2, suffix1, suffix2, calls_list):
    cm = np.zeros((num_labels + 1, num_labels + 1))
    for row_index in range(mat1.shape[0]):
        col_mat1_equal_one = np.where(mat1[row_index, :] == 1)[0]
        cols_mat2_equal_one = np.where(mat2[row_index, :] == 1)[0]
        if len(col_mat1_equal_one) == 1:
            if (
                mat2[row_index, col_mat1_equal_one] != -1
            ):  # only proceed if column in second matrix is not masked
                if len(cols_mat2_equal_one) > 0:
                    for cp_i in cols_mat2_equal_one:
                        cm[col_mat1_equal_one, cp_i] += 1 / len(cols_mat2_equal_one)
                else:
                    cm[col_mat1_equal_one, num_labels] += 1
        if len(col_mat1_equal_one) == 0:
            if len(cols_mat2_equal_one) > 0:
                for cp_i in cols_mat2_equal_one:
                    cm[num_labels, cp_i] += 1 / len(cols_mat2_equal_one)
            else:
                cm[num_labels, num_labels] += 1
        if len(col_mat1_equal_one) > 1:
            print("WARNING: more than one 1 in row of matrix y_true_stacked_drop")

    # normalize confusion matrix
    row_sum = np.sum(cm, axis=1, keepdims=True)
    cm = cm / row_sum
    cm = np.around(cm, 3)
    col_names = [suffix2 + "_" + x for x in calls_list]
    col_names += [suffix2 + "_NOLABEL"]
    row_names = [suffix1 + "_" + x for x in calls_list]
    row_names += [suffix1 + "_NOLABEL"]
    misclassification_table = pd.DataFrame(cm)
    misclassification_table.columns = col_names
    misclassification_table.index = row_names
    misclassification_table["fraction_time"] = np.around(row_sum / sum(row_sum), 5)

    return misclassification_table


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

y_pred_batch_test = y_pred_batch
y_true_batch_test = y_true_batch


# %%
# Convert to binary and stack batch
# convert y_pred_batch to binary values
y_pred_binary_batch_test = (y_pred_batch_test >= 0.5).astype(int)
# stack y_true_batch and y_pred_binary_batch_test
y_true_stacked = stack_batch(y_true_batch)
y_pred_binary_stacked = stack_batch(y_pred_binary_batch_test)

# %%
# compute misclassification table true vs pred
# get mask for those rows where there is at most one 1 (i.e. avoid overlapping labels)
mask = get_mask_for_rows_with_atmost_one_1(y_true_stacked)
# drop rows with overlapping labels in y_true_stacked
y_true_stacked_drop = y_true_stacked[mask]
y_pred_binary_stacked_drop = y_pred_binary_stacked[mask]
# compute misclassification
misclassification_table_true_vs_pred = compute_misclassification_table(
    y_true_stacked_drop,
    y_pred_binary_stacked_drop,
    "true",
    "pred",
    calls_for_labeling_list,
)
misclassification_table_true_vs_pred

# %%
# compute misclassification table pred vs true
# get mask for those rows where there is at most one 1 (i.e. avoid overlapping labels)
mask = get_mask_for_rows_with_atmost_one_1(y_pred_binary_stacked)
# drop rows with overlapping labels in y_true_stacked
y_true_stacked_drop = y_true_stacked[mask]
y_pred_binary_stacked_drop = y_pred_binary_stacked[mask]
misclassification_table_pred_vs_true = compute_misclassification_table(
    y_pred_binary_stacked_drop,
    y_true_stacked_drop,
    "pred",
    "true",
    calls_for_labeling_list,
)
misclassification_table_pred_vs_true


# %%
# run model on test part of extracted_snippets
test_all_df = extracted_snippets[["fnstem_path", "row_start", "row_stop"]][
    extracted_snippets["type"] == "test"
]
# test_all_df = test_all_df.sample(n=50000, replace=False).reset_index()
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
# Evaluate on all snippets rather than just test
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
print("     - # snippets:", len_test_data)
for spectrogram_batch, label_batch in test_all_dataset:
    print(".", sep="")
    if i % 80 == 0:
        print("")
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

y_pred_batch_all = y_pred_batch
y_true_batch_all = y_true_batch

# %%
# Convert to binary and stack batch
# convert y_pred_batch to binary values
y_pred_binary_batch_all = (y_pred_batch_all >= 0.5).astype(int)
# stack y_true_batch and y_pred_binary_batch_all
y_true_stacked = stack_batch(y_true_batch_all)
y_pred_binary_stacked = stack_batch(y_pred_binary_batch_all)

# %%
# compute misclassification table true vs pred
# get mask for those rows where there is at most one 1 (i.e. avoid overlapping labels)
mask = get_mask_for_rows_with_atmost_one_1(y_true_stacked)
# drop rows with overlapping labels in y_true_stacked
y_true_stacked_drop = y_true_stacked[mask]
y_pred_binary_stacked_drop = y_pred_binary_stacked[mask]
# compute misclassification
misclassification_table_all_true_vs_pred = compute_misclassification_table(
    y_true_stacked_drop,
    y_pred_binary_stacked_drop,
    "true",
    "pred",
    calls_for_labeling_list,
)
misclassification_table_all_true_vs_pred

# %%
# compute misclassification table pred vs true
# get mask for those rows where there is at most one 1 (i.e. avoid overlapping labels)
mask = get_mask_for_rows_with_atmost_one_1(y_pred_binary_stacked)
# drop rows with overlapping labels in y_true_stacked
y_true_stacked_drop = y_true_stacked[mask]
y_pred_binary_stacked_drop = y_pred_binary_stacked[mask]
misclassification_table_all_pred_vs_true = compute_misclassification_table(
    y_pred_binary_stacked_drop,
    y_true_stacked_drop,
    "pred",
    "true",
    calls_for_labeling_list,
)
misclassification_table_all_pred_vs_true


# %%
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
# show spec, lab_true, lab_pred for random element of test_df
show_random_snippets = False
if show_random_snippets:
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
