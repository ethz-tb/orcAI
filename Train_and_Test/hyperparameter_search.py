#!/usr/bin/env python

# %%
# import
import time
import sys
import os


import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
import keras_tuner as kt


# import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import auxiliary as aux
import model as mod
import load


# %%
# Read command line if interactive


aux.print_memory_usage()
interactive = aux.check_interactive()

if not interactive:
    (
        computer,
        model_name,
        data_dir,
        project_dir,
    ) = aux.hyperparameter_search_commandline_parse()
else:
    model_name = "cnn_res_lstm_model"
    computer = "laptop"
    data_dir = "/Users/sb/AI_data/tvtdata/"
    project_dir = "/Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/"


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
    "hyperparameter_dict": project_dir
    + "Results/"
    + model_name
    + "/hyperparameter.dict",
}
for key, value in dicts.items():
    print("  - reading", key)
    globals()[key] = aux.read_dict(value, True)

# %%
# load data sets from local  disk
print("Loading train, val and test datasets from disk:", data_dir)
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
input_shape = tuple(spectrogram.shape[1:])  #  shape
num_labels = labels.shape[2]  # Number of sound types


# %%
# Hyperparameter search
print("Hyperparameter search:")
for key, value in hyperparameter_dict.items():
    print("  - ", key, ":", value)

# %%
# Sequential hyperparameter search
if not hyperparameter_dict["multi_gpu"]:
    print("  - Sequential: # gpus", 1)
    # Metric
    masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
        fn=lambda y_true, y_pred: mod.masked_binary_accuracy(
            y_true, y_pred, mask_value=-1.0
        ),
        name="masked_binary_accuracy",
    )

    def model_builder(hp):
        hp_filters = hp.Choice(
            "filters", values=list(hyperparameter_dict["filters"].keys())
        )
        filters = hyperparameter_dict["filters"][hp_filters]
        hp_lstm_units = hp.Choice("lstm_units", hyperparameter_dict["lstm_units"])
        hp_dropout = hp.Choice("dropout_rate", hyperparameter_dict["dropout_rate"])
        hp_kernelsize = hp.Choice("kernel_size", hyperparameter_dict["kernel_size"])
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        model = mod.build_cnn_res_lstm_model(
            input_shape, num_labels, filters, hp_kernelsize, hp_dropout, hp_lstm_units
        )
        model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: mod.masked_binary_crossentropy(
                y_true, y_pred, mask_value=-1.0
            ),
            metrics=[masked_binary_accuracy_metric],
        )
        return model

    tuner = kt.Hyperband(
        model_builder,
        objective=kt.Objective(
            "val_masked_binary_accuracy", direction="max"
        ),  # Specify the objective explicitly
        max_epochs=10,
        directory=project_dir + "Results/" + model_dict["name"] + "/hp_logs",
        project_name=hyperparameter_dict["project_name"],
    )

    early_stopping = EarlyStopping(
        monitor="val_masked_binary_accuracy",  # Use the validation metric
        patience=5,  # Number of epochs to wait for improvement
        mode="max",  # Stop when accuracy stops increasing
        restore_best_weights=True,  # Restore weights from the best epoch
    )
    model_checkpoint = ModelCheckpoint(
        model_dict["name"],
        monitor="val_masked_binary_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        callbacks=[early_stopping, model_checkpoint],
        verbose=2,
    )


# %%
# Parallel hyperparameter search
start_time = time.time()
if hyperparameter_dict["multi_gpu"]:
    gpus = tf.config.list_physical_devices("GPU")
    print("  - Paralell: # gpus", len(gpus))

    def model_builder(hp):
        hp_filters = hp.Choice(
            "filters", values=list(hyperparameter_dict["filters"].keys())
        )
        filters = hyperparameter_dict["filters"][hp_filters]
        hp_lstm_units = hp.Choice("lstm_units", hyperparameter_dict["lstm_units"])
        hp_dropout = hp.Choice("dropout_rate", hyperparameter_dict["dropout_rate"])
        hp_kernelsize = hp.Choice("kernel_size", hyperparameter_dict["kernel_size"])
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
            fn=lambda y_true, y_pred: mod.masked_binary_accuracy(
                y_true, y_pred, mask_value=-1.0
            ),
            name="masked_binary_accuracy",
        )

        model = mod.build_cnn_res_lstm_model(
            input_shape, num_labels, filters, hp_kernelsize, hp_dropout, hp_lstm_units
        )
        model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: mod.masked_binary_crossentropy(
                y_true, y_pred, mask_value=-1.0
            ),
            metrics=[masked_binary_accuracy_metric],
        )
        return model

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        tuner = kt.Hyperband(
            model_builder,
            objective=kt.Objective("val_masked_binary_accuracy", direction="max"),
            max_epochs=10,
            directory=project_dir + "Results/" + model_dict["name"] + "/hp_logs",
            project_name=hyperparameter_dict["project_name"],
            executions_per_trial=1,
        )

    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        callbacks=[early_stopping, model_checkpoint],
        verbose=2,
    )
# %%

# %%
print(f"  - time for hyperparameter search: {time.time() - start_time:.2f} seconds")
print("PROGRAM COMPLETED")
