from functools import partial
from importlib.resources import files
from pathlib import Path

import keras
import keras_tuner as kt
import tensorflow as tf

from orcAI.architectures import (
    MaskedBinaryAccuracy,
    MaskedBinaryCrossentropy,
    build_model,
)
from orcAI.auxiliary import SEED_ID_LOAD_TEST_DATA, SEED_ID_LOAD_VAL_DATA, Messenger
from orcAI.io import load_dataset, read_json, write_json

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)


def _hp_model_builder(
    hp: kt.HyperParameters,
    input_shape: tuple[int, int, int],
    orcai_parameter: dict,
    hps_parameter: dict,
    msgr: Messenger = Messenger(verbosity=0),
) -> keras.Model:
    """Build a model for hyperparameter search
    Parameters
    ----------
    hp : kt.HyperParameters
        Hyperparameters for the model
    input_shape : tuple (int, int, int)
        Dimensions of the input data
    num_labels : int
        Number of labels to predict
    orcai_parameter : dict
        orcai parameters
    hps_parameter : dict
        Hyperparameter search parameters
    msgr : Messenger
        Messenger object for logging

    Returns
    -------
    keras.Model
        Model for hyperparameter search
    """
    hp_filters = hp.Choice("filters", values=list(hps_parameter["filters"].keys()))
    orcai_parameter["model"]["filters"] = hps_parameter["filters"][hp_filters]
    orcai_parameter["model"]["kernel_size"] = hp.Choice(
        "kernel_size", hps_parameter["kernel_size"]
    )
    orcai_parameter["model"]["dropout_rate"] = hp.Choice(
        "dropout_rate", hps_parameter["dropout_rate"]
    )
    orcai_parameter["model"]["batch_size"] = hp.Choice(
        "batch_size", hps_parameter["batch_size"]
    )
    if "lstm_units" in orcai_parameter["model"].keys():
        if "lstm_units" in hps_parameter.keys():
            orcai_parameter["model"]["lstm_units"] = hp.Choice(
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

    model = build_model(input_shape, orcai_parameter, msgr=msgr)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=orcai_parameter["model"]["learning_rate"]
        ),
        loss=MaskedBinaryCrossentropy(),
        metrics=[MaskedBinaryAccuracy()],
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
    data_compression: str | None = "GZIP",
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> None:
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
    save_best_model : bool
        Save the best model to the output directory
    data_compression: str | None
        Compression of data files. Accepts "GZIP" or "NONE".
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.

    Returns
    -------
    None
    """

    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Hyperparameter search")

    msgr.part("Loading Hyperparameter search parameter")

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)

    msgr.debug("Model parameter")
    msgr.debug(orcai_parameter["model"])
    model_name = orcai_parameter["name"]

    if isinstance(hps_parameter, (Path | str)):
        hps_parameter = read_json(hps_parameter)
    msgr.debug("Hyperparameter search parameter")
    msgr.debug(hps_parameter)

    msgr.part(f"Loading training and validation datasets from {data_dir}")
    dataset_shape = read_json(data_dir.joinpath("dataset_shapes.json"))
    train_dataset = load_dataset(
        data_dir.joinpath("train_dataset"),
        orcai_parameter["model"]["batch_size"],
        compression=data_compression,
        seed=[SEED_ID_LOAD_TEST_DATA, orcai_parameter["seed"]],
    )
    val_dataset = load_dataset(
        data_dir.joinpath("val_dataset"),
        orcai_parameter["model"]["batch_size"],
        compression=data_compression,
        seed=[SEED_ID_LOAD_VAL_DATA, orcai_parameter["seed"]],
    )

    msgr.part("Searching hyperparameters")
    hps_logs_dir = Path(output_dir).joinpath("hps_logs")

    if parallel:
        gpus = tf.config.list_physical_devices("GPU")
        msgr.info(f"Parallel - running on {len(gpus)} GPU")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            tuner = kt.Hyperband(
                partial(
                    _hp_model_builder,
                    input_shape=tuple(dataset_shape["spectrogram"]),
                    orcai_parameter=orcai_parameter,
                    hps_parameter=hps_parameter,
                ),
                objective=kt.Objective(
                    orcai_parameter["model"]["monitor"], direction="max"
                ),
                max_epochs=10,
                directory=hps_logs_dir,
                project_name=model_name,
                executions_per_trial=1,
            )
    else:
        msgr.info("Sequential - running on 1 GPU")
        tuner = kt.Hyperband(
            partial(
                _hp_model_builder,
                input_shape=tuple(dataset_shape["spectrogram"]),
                orcai_parameter=orcai_parameter,
                hps_parameter=hps_parameter,
            ),
            objective=kt.Objective(
                orcai_parameter["model"]["monitor"], direction="max"
            ),
            max_epochs=10,
            directory=hps_logs_dir,
            project_name=model_name,
        )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=orcai_parameter["model"]["monitor"],
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=0 if verbosity < 3 else 1,
    )

    msgr.info(f"Saving best model to hps/{model_name}.keras")
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        str(Path(output_dir).joinpath(model_name, "hps", model_name + ".keras")),
        monitor=orcai_parameter["model"]["monitor"],
        save_best_only=True,
    )
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0 if verbosity < 3 else 1,
    )
    msgr.part("Best Hyperparameters")
    msgr.info(tuner.get_best_hyperparameters()[0].values)
    write_json(
        tuner.get_best_hyperparameters()[0].values,
        hps_logs_dir.joinpath("best_hyperparameters.json"),
    )

    msgr.success("Hyperparameter search completed")

    return
