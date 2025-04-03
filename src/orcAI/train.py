from pathlib import Path
from importlib.resources import files
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm.keras import TqdmCallback

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)

# import local
from orcAI.auxiliary import Messenger, SEED_ID_LOAD_TRAIN_DATA, SEED_ID_LOAD_VAL_DATA
from orcAI.architectures import (
    build_model,
    masked_binary_accuracy,
    masked_binary_crossentropy,
)
from orcAI.io import load_dataset, read_json, write_json


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
    load_weights: bool = False,
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
    load_weights : bool
        Load weights from previous training.
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
    dataset_shape = read_json(data_dir.joinpath("dataset_shapes.json"))
    train_dataset = load_dataset(
        data_dir.joinpath("train_dataset"),
        model_parameter["batch_size"],
        compression="GZIP",
        seed=[SEED_ID_LOAD_TRAIN_DATA, orcai_parameter["seed"]],
    )
    val_dataset = load_dataset(
        data_dir.joinpath("val_dataset"),
        model_parameter["batch_size"],
        compression="GZIP",
        seed=[SEED_ID_LOAD_VAL_DATA, orcai_parameter["seed"]],
    )

    msgr.info(f"Batch size {model_parameter['batch_size']}")

    msgr.part("Building model")
    model = build_model(
        tuple(dataset_shape["spectrogram"]),
        orcai_parameter,
        msgr=msgr,
    )

    # Compiling Model
    msgr.part("Compiling model: " + model_name)

    # Loading model weights if required
    weights_path = output_dir.joinpath(model_name, model_name + ".weights.h5")
    if load_weights:
        msgr.info("Loading weights from stored model: " + model_name)
        model.load_weights(weights_path)
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
        patience=model_parameter["patience"],
        mode="max",
        restore_best_weights=True,
        verbose=0 if verbosity < 3 else 1,
    )
    model_checkpoint = ModelCheckpoint(
        weights_path,
        monitor="val_masked_binary_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=0 if verbosity < 3 else 1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_masked_binary_accuracy",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
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

    with tf.device("/GPU:0"):
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=model_parameter["epochs"],
            shuffle=True,
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
            verbose=0,
        )

    msgr.info(f"training history: {history.history}")
    msgr.part("Saving Model")

    with open(output_dir.joinpath(model_name, "training_history.json"), "w") as f:
        f.write(str(history.history))

    write_json(
        orcai_parameter,
        output_dir.joinpath(model_name).joinpath(f"orcai_parameter.json"),
    )
    model.save_weights(weights_path)
    model.save(
        output_dir.joinpath(model_name, model_name + ".keras"),
        include_optimizer=True,
    )
    model_shape = {
        "input_shape": dataset_shape["spectrogram"],
        "num_labels": len(label_calls),
    }
    write_json(model_shape, output_dir.joinpath(model_name, "model_shape.json"))

    msgr.success(f"Training model finished. Model saved to {model_name + '.keras'}")
    return
