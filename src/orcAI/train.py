import time
from pathlib import Path
from importlib.resources import files
from functools import partial
import numpy as np
import click
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras_tuner as kt

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)

# import local
from orcAI.auxiliary import Messenger, read_json, write_json
from orcAI.architectures import (
    build_model,
    masked_binary_accuracy,
    masked_binary_crossentropy,
)
from orcAI.load import reload_dataset


# model parameters
def _count_params(trainable_weights: list) -> int:
    """Count the number of trainable parameters in a model
    Parameters
    ----------
    trainable_weights : list
        List of trainable weights in the model
    Returns
    -------
    int
        Number of parameters in the model
    """
    return np.sum([np.prod(w.shape) for w in trainable_weights])


def train(
    data_dir,
    output_dir,
    orcai_parameter=files("orcAI.defaults").joinpath("default_orcai_parameter.json"),
    load_weights=False,
    verbosity=1,
):
    """Trains an orcAI model

    Parameters
    ----------
    data_dir : Path | str
        Path to the directory containing the training, validation and test datasets.
    output_dir : Path | str
        Path to the output directory.
    orcai_parameter : (Path | str) | dict
        Path to a JSON file containing orcai parameter or a dictionary with orcai parameter.
    load_weights : bool
        Load weights from previous training.
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    """
    # Initialize messenger
    msgr = Messenger(verbosity=verbosity)

    msgr.part("OrcAI - training model")
    msgr.info(f"Output directory: {output_dir}")
    msgr.info(f"Data directory: {data_dir}")

    msgr.info("Loading parameter and data...", indent=1)
    msgr.debug("Model parameter")
    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)
    model_parameter = orcai_parameter["model"]
    label_calls = orcai_parameter["calls"]

    msgr.debug(model_parameter)
    model_name = model_parameter["name"]
    msgr.debug("Calls for labeling")
    msgr.debug(label_calls, indent=-1)

    file_paths = {
        "training_data": Path(data_dir).joinpath("train_dataset"),
        "validation_data": Path(data_dir).joinpath("val_dataset"),
        "test_data": Path(data_dir).joinpath("test_dataset"),
        "model": Path(output_dir).joinpath(model_name, model_name + ".h5"),
        "model_dir": Path(output_dir).joinpath(model_name),
        "weights": Path(output_dir).joinpath(model_name, model_name + ".weights.h5"),
        "history": Path(output_dir).joinpath(model_name, "training_history.json"),
        "confusion_matrices": Path(output_dir).joinpath("confusion_matrices.json"),
    }

    # load data sets from local disk
    msgr.info(f"Loading train, val and test datasets from {data_dir}", indent=1)
    tf.config.set_soft_device_placement(True)
    start_time = time.time()
    train_dataset = reload_dataset(
        file_paths["training_data"], model_parameter["batch_size"]
    )
    val_dataset = reload_dataset(
        file_paths["validation_data"], model_parameter["batch_size"]
    )
    test_dataset = reload_dataset(
        file_paths["test_data"], model_parameter["batch_size"]
    )
    msgr.info(f"time to load datasets: {time.time() - start_time:.2f} seconds")

    # Verify the val dataset and obtain shape
    spectrogram, labels = val_dataset.take(1).element_spec
    msgr.info(f"Spectrogram batch shape: {spectrogram.shape}")
    msgr.info(f"Labels batch shape: {labels.shape}")

    # Build model architecture
    input_shape = tuple(spectrogram.shape[1:])  # shape
    num_labels = labels.shape[2]  # Number of sound types

    model = build_model(input_shape, num_labels, orcai_parameter, msgr=msgr)

    # Compiling Model
    # Loading model weights if required
    msgr.part("Compiling model: " + model_name)
    if load_weights:
        msgr.info("Loading weights from stored model: " + model_name)
        model.load_weights(file_paths["weights"])
    else:
        msgr.info("Learning weights from scratch")

    # Metrics
    masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
        fn=masked_binary_accuracy,
        name="masked_binary_accuracy",
    )

    model.compile(
        optimizer="adam",
        loss=masked_binary_crossentropy,
        metrics=[masked_binary_accuracy_metric],
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_masked_binary_accuracy",
        patience=model_parameter[
            "patience"
        ],  # Number of epochs to wait for improvement
        mode="max",  # Stop when accuracy stops increasing
        restore_best_weights=True,  # Restore weights from the best epoch
        verbose=0 if verbosity < 3 else 1,
    )
    model_checkpoint = ModelCheckpoint(
        file_paths["weights"],
        monitor="val_masked_binary_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=0 if verbosity < 3 else 1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_masked_binary_accuracy",
        factor=0.5,  # Reduce learning rate by a factor of 0.5
        patience=3,  # Wait for 3 epochs of no improvement
        min_lr=1e-6,  # Set a lower limit for the learning rate
        verbose=0 if verbosity < 3 else 1,  # Print updates to the console
    )

    total_params = model.count_params()
    trainable_params = _count_params(model.trainable_weights)
    non_trainable_params = _count_params(model.non_trainable_weights)
    msgr.info("Model size:", indent=1)
    msgr.info(f"Total parameters: {total_params}")
    msgr.info(f"Trainable parameters: {trainable_params}")
    msgr.info(f"Non-trainable parameters: {non_trainable_params}")
    msgr.print_memory_usage()

    # Train model
    msgr.part(f"Training model: {model_name}")
    start_time = time.time()

    with tf.device("/GPU:0"):
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=model_parameter["epochs"],
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1 if verbosity > 0 else 0,
        )
    msgr.info(f"total time for training: {time.time() - start_time:.2f} seconds")

    msgr.info(f"training history: {history.history}")
    msgr.info(f"saving training history: {file_paths['history']}")
    with open(file_paths["history"], "w") as f:
        f.write(str(history.history))

    write_json(
        orcai_parameter,
        file_paths["model_dir"].joinpath(f"orcai_parameter.json"),
    )
    write_json(
        {"input_shape": input_shape, "num_labels": num_labels},
        file_paths["model_dir"].joinpath("model_shape.json"),
    )
    # TODO: Save model?
    model.save(file_paths["model"], include_optimizer=True)

    msgr.success(
        f"OrcAI - training model finished. Model saved to {file_paths['model']}"
    )
    return


def _hp_model_builder(
    hp: kt.HyperParameters,
    input_shape: tuple[int, int, int],
    num_labels: int,
    model_parameter: dict,
    hps_parameter: dict,
    msgr: Messenger = Messenger(verbosity=0),
) -> tf.keras.Model:
    """Build a model for hyperparameter search
    Parameters
    ----------
    hp : kt.HyperParameters
        Hyperparameters for the model
    input_shape : tuple (int, int, int)
        Dimensions of the input data
    num_labels : int
        Number of labels to predict
    model_parameter : dict
        Model parameters
    hps_parameter : dict
        Hyperparameter search parameters
    msgr : Messenger
        Messenger object for logging

    Returns
    -------
    tf.keras.Model
        Model for hyperparameter search
    """
    hp_filters = hp.Choice("filters", values=list(hps_parameter["filters"].keys()))
    model_parameter["filters"] = hps_parameter["filters"][hp_filters]
    model_parameter["kernel_size"] = hp.Choice(
        "kernel_size", hps_parameter["kernel_size"]
    )
    model_parameter["dropout_rate"] = hp.Choice(
        "dropout_rate", hps_parameter["dropout_rate"]
    )
    if "lstm_units" in model_parameter.keys():
        if "lstm_units" in hps_parameter.keys():
            model_parameter["lstm_units"] = hp.Choice(
                "lstm_units", hps_parameter["lstm_units"]
            )
        else:
            raise ValueError(
                "LSTM units not in hyperparameter search parameters. "
                + "Is the right model specified?"
            )
    else:
        if "lstm_units" in hps_parameter.keys():
            raise ValueError(
                "LSTM units not in model parameters. Is the right model specified?"
            )

    model = build_model(input_shape, num_labels, model_parameter, msgr=msgr)

    masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
        fn=masked_binary_accuracy,
        name="masked_binary_accuracy",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=masked_binary_crossentropy,
        metrics=[masked_binary_accuracy_metric],
    )
    return model


def hyperparameter_search(
    data_dir: Path | str,
    output_dir: Path | str,
    orcai_parameter: Path | str = files("orcAI.defaults").joinpath(
        "default_orcai_parameter.json"
    ),
    hps_parameter: Path | str = files("orcAI.defaults").joinpath(
        "default_hps_parameter.json"
    ),
    parallel: bool = False,
    verbosity: int = 2,
):
    """Perform hyperparameter search
    Parameters
    ----------
    data_dir : Path | str
        Path to the data directory
    output_dir : Path | str
        Path to the output directory
    orcai_parameter : Path | str
        Path to the OrcAi parameter file
    hps_parameter : Path | str
        Path to the hyperparameter search parameter file
    parallel : bool
        Run hyperparameter search on multiple GPUs
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug

    Returns
    -------
    None
    """

    msgr = Messenger(verbosity=verbosity)

    msgr.part("Running Hyperparameter Search")
    msgr.info(f"Data directory: {data_dir}")

    msgr.info("Loading parameter and data...", indent=1)
    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)
    model_parameter = orcai_parameter["model"]
    msgr.debug("Model parameter")
    msgr.debug(model_parameter)
    model_name = model_parameter["name"]

    msgr.info("Loading hyperparameter search parameter...", indent=1)
    if isinstance(hps_parameter, (Path | str)):
        hps_parameter = read_json(hps_parameter)
    msgr.debug("Hyperparameter search parameter")
    msgr.debug(hps_parameter)

    file_paths = {
        "training_data": Path(data_dir).joinpath("train_dataset"),
        "validation_data": Path(data_dir).joinpath("val_dataset"),
        "test_data": Path(data_dir).joinpath("test_dataset"),
        "model_dir": Path(output_dir).joinpath(model_name),
        "hps_dir": Path(output_dir).joinpath(model_name, "hps"),
        "hps_logs_dir": Path(output_dir).joinpath(model_name, "hps_logs"),
    }

    # load data sets from local disk
    msgr.info(f"Loading train and val datasets from {data_dir}", indent=1)
    start_time = time.time()
    train_dataset = reload_dataset(
        file_paths["training_data"], model_parameter["batch_size"]
    )
    val_dataset = reload_dataset(
        file_paths["validation_data"], model_parameter["batch_size"]
    )
    msgr.info(f"time to load datasets: {time.time() - start_time:.2f} seconds")

    # Verify the val dataset and obtain shape
    spectrogram, labels = val_dataset.take(1).element_spec
    msgr.info(f"Spectrogram batch shape: {spectrogram.shape}")
    msgr.info(f"Labels batch shape: {labels.shape}")

    input_shape = tuple(spectrogram.shape[1:])  #  shape
    num_labels = labels.shape[2]  # Number of sound types

    if parallel:
        gpus = tf.config.list_physical_devices("GPU")
        msgr.info(f"Prallel - running on {len(gpus)} GPU")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            tuner = kt.Hyperband(
                partial(
                    _hp_model_builder,
                    input_shape=input_shape,
                    num_labels=num_labels,
                    model_parameter=model_parameter,
                    hps_parameter=hps_parameter,
                ),
                objective=kt.Objective("val_masked_binary_accuracy", direction="max"),
                max_epochs=10,
                directory=file_paths["hps_logs_dir"],
                project_name=model_name,
                executions_per_trial=1,
            )
    else:
        msgr.info(f"Sequential - running on 1 GPU")
        tuner = kt.Hyperband(
            partial(
                _hp_model_builder,
                input_shape=input_shape,
                num_labels=num_labels,
                model_parameter=model_parameter,
                hps_parameter=hps_parameter,
            ),
            objective=kt.Objective("val_masked_binary_accuracy", direction="max"),
            max_epochs=10,
            directory=file_paths["hps_logs_dir"],
            project_name=model_name,
        )
    early_stopping = EarlyStopping(
        monitor="val_masked_binary_accuracy",  # Use the validation metric
        patience=5,  # Number of epochs to wait for improvement TODO: different than for train, intentional?
        mode="max",  # Stop when accuracy stops increasing
        restore_best_weights=True,  # Restore weights from the best epoch
        verbose=0 if verbosity < 3 else 1,
    )
    model_checkpoint = ModelCheckpoint(
        file_paths["hps_dir"].joinpath(model_name + ".weights.h5"),
        monitor="val_masked_binary_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        callbacks=[early_stopping, model_checkpoint],
        verbose=verbosity,
    )
    msgr.info(f"Time for hyperparameter search: {time.time() - start_time:.2f} seconds")
    msgr.success("Hyperparameter search completed")
    return
