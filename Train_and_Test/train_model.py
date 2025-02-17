#!/usr/bin/env python
# %%
# import
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os


from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.backend import count_params


# import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import auxiliary as aux
import model as mod
import load as load


# %%
# Read command line if interactive

interactive = aux.check_interactive()
log_file = None

if not interactive:
    print("Command-line call:", " ".join(sys.argv))
    (computer, load_weights, data_dir, model_name, project_dir) = (
        aux.train_model_commandline_parse()
    )
else:
    load_weights = False
    data_dir = "/Users/sb/AI_data/tvtdata/"
    model_name = "cnn_res_model"

if not data_dir.endswith("/"):
    data_dir = data_dir + "/"
# %%
# Read parameters
print("Project directory:", project_dir)
os.chdir(project_dir)

print("READ IN PARAMETERS")
dicts = {
    "directories_dict": "GenericParameters/directories.dict",
    "call_dict": "GenericParameters/call.dict",
    "spectrogram_dict": "GenericParameters/spectrogram.dict",
    "model_dict": project_dir + "Results/" + model_name + "/model.dict",
    "calls_for_labeling_list": "GenericParameters/calls_for_labeling.list",
}
for key, value in dicts.items():
    print("  - reading", key)
    globals()[key] = aux.read_dict(value, True)


# %%
# load data sets from local  disk
print("Loading train, val and test datasets from disk:", data_dir)
tf.config.set_soft_device_placement(True)
start_time = time.time()
train_dataset = load.reload_dataset(
    data_dir + "train_dataset", model_dict["batch_size"]
)
val_dataset = load.reload_dataset(data_dir + "val_dataset", model_dict["batch_size"])
test_dataset = load.reload_dataset(data_dir + "test_dataset", model_dict["batch_size"])
print(f"  - time to load datasets: {time.time() - start_time:.2f} seconds")


# %%
# Verify the val dataset and obtain shape
for spectrogram, labels in val_dataset.take(1):
    print("Spectrogram batch shape:", spectrogram.numpy().shape)
    print("Labels batch shape:", labels.numpy().shape)

# %%
# Build Model
print("Build model:")
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
input_shape = tuple(spectrogram.shape[1:])  #  shape
num_labels = labels.shape[2]  # Number of sound types

model = mod.build_model(model_choice_dict, model_dict, input_shape, num_labels)


# %% TRANSFORMER MODEL FIX
transformer_parallel = False
if transformer_parallel:
    if model_dict["name"] == "cnn_res_transformer_model":
        # Define model within strategy scope
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
                fn=lambda y_true, y_pred: mod.masked_binary_accuracy(
                    y_true, y_pred, mask_value=-1.0
                ),
                name="masked_binary_accuracy",
            )
            model = mod.build_cnn_res_transformer_model(
                input_shape,
                num_labels,
                filters,
                kernel_size,
                dropout_rate,
                num_heads,
            )
            model.compile(
                optimizer="adam",
                loss=lambda y_true, y_pred: mod.masked_binary_crossentropy(
                    y_true, y_pred, mask_value=-1.0
                ),
                metrics=[masked_binary_accuracy_metric],
            )

# %%
# Loading model weights if required
print("Fitting mode:", model_dict["name"])
if load_weights:
    print("  - Loading weights from stored model:", model_dict["name"])
    model.load_weights(
        project_dir + "Results/" + model_dict["name"] + "/" + model_dict["name"]
    )
else:
    print("  - Learning weights from scratch")

# %%
# Compiling Model
print("Compiling model:", model_dict["name"])
# Metrics
masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
    fn=lambda y_true, y_pred: mod.masked_binary_accuracy(
        y_true, y_pred, mask_value=-1.0
    ),
    name="masked_binary_accuracy",
)
masked_f1_metric = tf.keras.metrics.MeanMetricWrapper(
    fn=lambda y_true, y_pred: masked_f1_score(
        y_true, y_pred, mask_value=-1.0, threshold=0.5
    ),
    name="masked_f1_score",
)
# Callbacks
early_stopping = EarlyStopping(
    monitor="val_masked_binary_accuracy",  # swap val_masked_binary_accuracy with val_masked_f1_score as needed
    patience=model_dict["patience"],  # Number of epochs to wait for improvement
    mode="max",  # Stop when accuracy stops increasing
    restore_best_weights=True,  # Restore weights from the best epoch
)
model_checkpoint = ModelCheckpoint(
    project_dir + "Results/" + model_dict["name"] + "/" + model_dict["name"],
    monitor="val_masked_binary_accuracy",  # swap val_masked_binary_accuracy with val_masked_f1_score as needed
    save_best_only=True,
    save_weights_only=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_masked_binary_accuracy",  # swap val_masked_binary_accuracy with val_masked_f1_score as needed
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
    metrics=[masked_binary_accuracy_metric, masked_f1_metric],
)


# model parameters
def count_params(trainable_weights):
    return np.sum([np.prod(w.shape) for w in trainable_weights])


total_params = model.count_params()
trainable_params = count_params(model.trainable_weights)
non_trainable_params = count_params(model.non_trainable_weights)
print("Model size:")
print(f"  - Total parameters: {total_params}")
print(f"  - Trainable parameters: {trainable_params}")
print(f"  - Non-trainable parameters: {non_trainable_params}")
aux.print_memory_usage()

# %%
# Train model
print("Training model:", model_dict["name"])
start_time = time.time()
if computer == "euler":
    verbosity = 2
else:
    verbosity = 1
with tf.device("/GPU:0"):
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=model_dict["epochs"],
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=verbosity,
    )
print(f"  - total time for training: {time.time() - start_time:.2f} seconds")
history_dict_file = (
    project_dir + "Results/" + model_dict["name"] + "/training_history.dict"
)
print("  - training history dictionary:", history.history)
print("  - saving training history:", history_dict_file)
with open(history_dict_file, "w") as f:
    f.write(str(history.history))


# %%
# Model evaluation
print("Evaluate model:", model_dict["name"])
test_loss, test_metric = model.evaluate(test_dataset)
print(f"  - test loss: {test_loss}")
print(f"  - test masked binary accuracy: {test_metric}")

print(f"  - confusion matrices:")
# Extract true labels
y_pred_batch = []
y_true_batch = []
i = 1
len_test_data = len(test_dataset)
print("Predicting test data:")
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
mod.masked_binary_accuracy(y_true_batch, y_pred_batch, mask_value=-1.0)
aux.write_dict(
    confusion_matrices,
    project_dir + "Results/" + model_dict["name"] + "/" + "/confusion_matrices",
)


# %%
print("PROGRAM COMPLETED")

# %%
