from importlib.resources import files
from pathlib import Path

import tf_keras as keras
import numpy as np
import tensorflow as tf
from tqdm.keras import TqdmCallback

from orcAI.architectures import (
    MaskedAUC,
    MaskedBinaryAccuracy,
    MaskedBinaryCrossentropy,
    build_model,
)

# import local
from orcAI.auxiliary import SEED_ID_LOAD_TRAIN_DATA, SEED_ID_LOAD_VAL_DATA, Messenger
from orcAI.io import load_dataset, load_orcai_model, read_json, write_json

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)


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
    data_dir: Path | str,
    output_dir: Path | str,
    orcai_parameter: (Path | str) | dict = files("orcAI.defaults").joinpath(
        "default_orcai_parameter.json"
    ),
    data_compression: str | None = "GZIP",
    load_model: bool = False,
    verbosity: int = 2,
    msgr: Messenger | None = None,
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
    data_compression: str | None
        Compression of data files. Accepts "GZIP" or "NONE".
    load_model : bool
        Load model from previous training.
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    """
    if msgr is None:
        msgr = Messenger(
            verbosity=verbosity,
            title="Training model",
        )
    msgr.print_platform_info(set_indent=1)
    msgr.print_tf_device_info(set_indent=1)

    msgr.part("Loading parameter")

    msgr.info(f"Output directory: {output_dir}")
    msgr.info(f"Data directory: {data_dir}")
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)

    msgr.debug("Model parameter")
    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)
    model_name = orcai_parameter["name"]
    model_parameter = orcai_parameter["model"]
    label_calls = orcai_parameter["calls"]

    msgr.debug(model_parameter)
    msgr.debug("Calls for labeling")
    msgr.debug(label_calls, indent=-1)

    # load data sets from local disk
    msgr.part(f"Loading training and validation datasets from {data_dir}")
    tf.config.set_soft_device_placement(True)
    if data_dir.joinpath("dataset_shapes.json").exists():
        msgr.info("Loading dataset shapes from JSON file")
        dataset_shape = read_json(data_dir.joinpath("dataset_shapes.json"))
    else:
        msgr.info("Using default OrcAI dataset shapes")
        dataset_shape = {"spectrogram": [736, 171, 1], "labels": [46, 7]}

    train_dataset = load_dataset(
        data_dir.joinpath("train_dataset"),
        model_parameter["batch_size"],
        compression=data_compression,
        seed=[SEED_ID_LOAD_TRAIN_DATA, orcai_parameter["seed"]],
    )
    val_dataset = load_dataset(
        data_dir.joinpath("val_dataset"),
        model_parameter["batch_size"],
        compression=data_compression,
        seed=[SEED_ID_LOAD_VAL_DATA, orcai_parameter["seed"]],
    )
    if orcai_parameter["model"]["call_weights"] is not None:
        call_weights = read_json(
            data_dir.joinpath("call_weights.json"),
        )
        msgr.info(f"Call weights: {call_weights}")
        if list(call_weights.keys()) != label_calls:
            raise ValueError(
                "Call weights do not match label calls. Please check the call weights file. Order of calls must be the same as in the orcAI parameter file."
            )
        call_weights_int = {n: call_weights[key] for n, key in enumerate(call_weights)}
    else:
        call_weights_int = None

    msgr.info(f"Batch size {model_parameter['batch_size']}")
    model_dir = output_dir.joinpath(model_name)

    if load_model:
        msgr.part("Loading model")
        model, _, _ = load_orcai_model(model_dir)
    else:
        msgr.part("Building model")
        model = build_model(
            tuple(dataset_shape["spectrogram"]),
            orcai_parameter,
            msgr=msgr,
        )

        # Compiling Model
        msgr.part("Compiling model: " + model_name)

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=model_parameter["learning_rate"]
            ),
            loss=MaskedBinaryCrossentropy(),
            metrics=[MaskedAUC(), MaskedBinaryAccuracy()],
        )

    # Callbacks

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=model_parameter["monitor"],
        patience=model_parameter["EarlyStopping_patience"],
        mode="max",
        restore_best_weights=True,
        verbose=0 if verbosity < 3 else 1,
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        model_dir.joinpath(model_name + ".keras"),
        monitor=model_parameter["monitor"],
        save_best_only=True,
        verbose=0 if verbosity < 3 else 1,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=model_parameter["monitor"],
        factor=model_parameter["ReduceLROnPlateau_factor"],
        patience=model_parameter["ReduceLROnPlateau_patience"],
        min_lr=model_parameter["min_learning_rate"],
        verbose=0 if verbosity < 3 else 1,
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
    msgr.part(f"Fitting model: {model_name}")
    msgr.info(f"Monitoring {model_parameter['monitor']}")

    with tf.device("/GPU:0"):
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=model_parameter["epochs"],
            callbacks=[
                early_stopping,
                model_checkpoint,
                reduce_lr,
                TqdmCallback(
                    data_size=model_parameter["batch_size"]
                    * model_parameter["n_batch_train"],
                    batch_size=model_parameter["batch_size"],
                    verbose=1 if verbosity > 2 else 0,
                ),
            ],
            steps_per_epoch=model_parameter["n_batch_train"],
            validation_steps=model_parameter["n_batch_val"],
            class_weight=call_weights_int,
            verbose=0,
        )

    msgr.part("Saving Model")

    model.save(
        model_dir.joinpath(model_name + ".keras"),
        include_optimizer=True,
    )

    write_json(
        history.history,
        model_dir.joinpath(model_name).joinpath("training_history.json"),
    )

    write_json(
        orcai_parameter,
        output_dir.joinpath(model_name).joinpath("orcai_parameter.json"),
    )

    model_shape = {
        "input_shape": dataset_shape["spectrogram"],
        "num_labels": len(label_calls),
    }
    write_json(model_shape, output_dir.joinpath(model_name, "model_shape.json"))

    msgr.success(f"Training model finished. Model saved to {model_name + '.keras'}")
    return
