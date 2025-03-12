import time
from pathlib import Path
from importlib.resources import files
from functools import partial
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
import keras_tuner as kt

from orcAI.auxiliary import Messenger, read_json
from orcAI.architectures import (
    res_net_LSTM_arch,
    masked_binary_accuracy,
    masked_binary_crossentropy,
)
from orcAI.load import reload_dataset


def _hp_model_builder(
    hp: kt.HyperParameters,
    input_shape: tuple[int, int, int],
    num_labels: int,
    model_parameter: dict,
    hps_parameter: dict,
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

    Returns
    -------
    tf.keras.Model
        Model for hyperparameter search
    """
    hp_filters = hp.Choice("filters", values=list(hps_parameter["filters"].keys()))
    filters = hps_parameter["filters"][hp_filters]
    hp_lstm_units = hp.Choice("lstm_units", hps_parameter["lstm_units"])
    hp_dropout_rate = hp.Choice("dropout_rate", hps_parameter["dropout_rate"])
    hp_kernel_size = hp.Choice("kernel_size", hps_parameter["kernel_size"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model = res_net_LSTM_arch(
        input_shape, num_labels, filters, hp_kernel_size, hp_dropout_rate, hp_lstm_units
    )

    masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
        fn=lambda y_true, y_pred: masked_binary_accuracy(
            y_true, y_pred, **model_parameter["masked_binary_accuracy_metric"]
        ),
        name="masked_binary_accuracy",
    )

    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: masked_binary_crossentropy(
            y_true, y_pred, mask_value=-1.0
        ),
        metrics=[masked_binary_accuracy_metric],
    )
    return model


def hyperparameter_search(
    data_dir: Path | str,
    output_dir: Path | str,
    model_parameter: Path | str = files("orcAI.defaults").joinpath(
        "default_model_parameter.json"
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
    model_parameter : Path | str
        Path to the model parameter file
    hps_parameter : Path | str
        Path to the hyperparameter search parameter file
    parallel : bool
        Run hyperparameter search on multiple GPUs
    verbosity : int
        Verbosity level.

    Returns
    -------
    None
    """

    msgr = Messenger(verbosity=verbosity)

    msgr.part("Running Hyperparameter Search")
    msgr.info(f"Data directory: {data_dir}")

    msgr.info("Loading parameter and data...", indent=1)
    if isinstance(model_parameter, (Path | str)):
        model_parameter = read_json(model_parameter)
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
    msgr.info(f"Loading train, val and test datasets from {data_dir}", indent=1)
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
            objective=kt.Objective(
                "val_masked_binary_accuracy", direction="max"
            ),  # Specify the objective explicitly
            max_epochs=10,
            directory=file_paths["hps_logs_dir"],
            project_name=model_name,
        )
    early_stopping = EarlyStopping(
        monitor=model_parameter["callback_metric"],  # Use the validation metric
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
