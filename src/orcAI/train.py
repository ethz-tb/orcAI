import time
from pathlib import Path
from importlib.resources import files
import numpy as np
import click

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# import local
from orcAI.auxiliary import Messenger, read_json, write_json, compute_confusion_matrix
from orcAI.architectures import (
    build_model,
    masked_binary_accuracy,
    masked_binary_crossentropy,
    masked_f1_score,
    build_cnn_res_transformer_arch,
)
from orcAI.load import reload_dataset


# model parameters
def count_params(trainable_weights):
    return np.sum([np.prod(w.shape) for w in trainable_weights])


def train(
    data_dir,
    output_dir,
    model_parameter=files("orcAI.defaults").joinpath("default_model_parameter.json"),
    label_calls=files("orcAI.defaults").joinpath("default_calls.json"),
    load_weights=False,
    transformer_parallel=False,
    verbosity=1,
):
    """Trains an orcAI model

    Parameters
    ----------
    data_dir : Path
        Path to the directory containing the training, validation and test datasets.
    output_dir : Path
        Path to the output directory.
    model_parameter_path : (Path | str) | dict
        Path to a JSON file containing model specifications or a dictionary with model specifications.
    label_calls : (Path | str) | dict
        Path to a JSON file containing calls to be labeled or a dictionary with calls to be labeled.
    load_weights : bool
        Load weights from previous training.
    transformer_parallel : bool
        Use transformer fix #TODO: Is this necessary
    verbosity : int
        Verbosity level.
    """
    # Initialize messenger
    msgr = Messenger(verbosity=verbosity)

    msgr.part("OrcAI - training model")
    msgr.info(f"Output directory: {output_dir}")
    msgr.info(f"Data directory: {data_dir}")

    msgr.info("Loading parameter and data...", indent=1)
    msgr.info("Model parameter")
    if isinstance(model_parameter, (Path | str)):
        model_parameter = read_json(model_parameter)

    msgr.info(model_parameter)
    model_name = model_parameter["name"]

    if isinstance(label_calls, (Path | str)):
        label_calls = read_json(label_calls)

    msgr.info("Calls for labeling")
    msgr.info(label_calls, indent=-1)

    # TODO: Model Weights, dca had to make a change here, because of Keras API changes
    file_paths = {
        "training_data": data_dir.joinpath("train_dataset"),
        "validation_data": data_dir.joinpath("val_dataset"),
        "test_data": data_dir.joinpath("test_dataset"),
        "model": output_dir.joinpath(model_name, model_name + ".h5"),
        "model_dir": output_dir.joinpath(model_name),
        "weights": output_dir.joinpath(model_name, model_name + ".weights.h5"),
        "history": output_dir.joinpath(model_name, "training_history.json"),
        "confusion_matrices": output_dir.joinpath("confusion_matrices.json"),
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
    for spectrogram, labels in val_dataset.take(1):
        msgr.info(f"Spectrogram batch shape: {spectrogram.numpy().shape}")
        msgr.info(f"Labels batch shape: {labels.numpy().shape}")

    # Build model architecture
    input_shape = tuple(spectrogram.shape[1:])  # shape
    num_labels = labels.shape[2]  # Number of sound types

    model = build_model(input_shape, num_labels, model_parameter, msgr=msgr)

    # TODO: is this necessary? if yes, rename to something more descriptive than fix
    # TRANSFORMER MODEL FIX
    if transformer_parallel:
        if model_name == "cnn_res_transformer_model":
            # Define model within strategy scope
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
                    fn=lambda y_true, y_pred: masked_binary_accuracy(
                        y_true, y_pred, mask_value=-1.0
                    ),
                    name="masked_binary_accuracy",
                )
                model = build_cnn_res_transformer_arch(
                    input_shape, num_labels, **model_parameter
                )
                model.compile(
                    optimizer="adam",
                    loss=lambda y_true, y_pred: masked_binary_crossentropy(
                        y_true, y_pred, mask_value=-1.0
                    ),
                    metrics=[masked_binary_accuracy_metric],
                )

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
        fn=lambda y_true, y_pred: masked_binary_accuracy(
            y_true, y_pred, **model_parameter["masked_binary_accuracy_metric"]
        ),
        name="masked_binary_accuracy",
    )
    masked_f1_metric = tf.keras.metrics.MeanMetricWrapper(
        fn=lambda y_true, y_pred: masked_f1_score(
            y_true, y_pred, **model_parameter["masked_f1_metric"]
        ),
        name="masked_f1_score",
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor=model_parameter[
            "callback_metric"
        ],  # val_masked_binary_accuracy | val_masked_f1_score
        patience=model_parameter[
            "patience"
        ],  # Number of epochs to wait for improvement
        mode="max",  # Stop when accuracy stops increasing
        restore_best_weights=True,  # Restore weights from the best epoch
        verbose=verbosity,
    )
    model_checkpoint = ModelCheckpoint(
        file_paths["weights"],
        monitor=model_parameter[
            "callback_metric"
        ],  # val_masked_binary_accuracy | val_masked_f1_score
        save_best_only=True,
        save_weights_only=True,
        verbose=1 if verbosity > 0 else 0,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor=model_parameter[
            "callback_metric"
        ],  # val_masked_binary_accuracy | val_masked_f1_score
        factor=0.5,  # Reduce learning rate by a factor of 0.5
        patience=3,  # Wait for 3 epochs of no improvement
        min_lr=1e-6,  # Set a lower limit for the learning rate
        verbose=1 if verbosity > 0 else 0,  # Print updates to the console
    )
    model.compile(
        optimizer="adam",
        loss=lambda y_true, y_pred: masked_binary_crossentropy(
            y_true, y_pred, mask_value=-1.0
        ),
        metrics=[masked_binary_accuracy_metric, masked_f1_metric],
    )

    total_params = model.count_params()
    trainable_params = count_params(model.trainable_weights)
    non_trainable_params = count_params(model.non_trainable_weights)
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

    # Model evaluation
    msgr.info(f"Evaluating model: {model_name}", level="part")
    test_loss, test_metric = model.evaluate(test_dataset)
    msgr.info(f"test loss: {test_loss}")
    msgr.info(f"test masked binary accuracy: {test_metric}")

    msgr.part("Predicting test data:")
    # Extract true labels
    y_pred_batch = []
    y_true_batch = []

    with click.progressbar(test_dataset, label="Predicting test data") as test_data:
        for spectrogram_batch, label_batch in test_data:
            y_true_batch.append(label_batch.numpy())
            y_pred_batch.append(model.predict(spectrogram_batch, verbose=0))

    y_true_batch = np.concatenate(y_true_batch, axis=0)
    y_pred_batch = np.concatenate(y_pred_batch, axis=0)
    confusion_matrices = compute_confusion_matrix(
        y_true_batch, y_pred_batch, label_calls, mask_value=-1
    )
    msgr.info(f"confusion matrices:", indent=1)
    msgr.print_confusion_matrices(confusion_matrices)
    masked_binary_accuracy(y_true_batch, y_pred_batch, mask_value=-1.0)
    write_json(confusion_matrices, file_paths["confusion_matrices"])
    msgr.print_dict(confusion_matrices)

    write_json(
        model_parameter, file_paths["model_dir"].joinpath("model_parameter.json")
    )
    write_json(label_calls, file_paths["model_dir"].joinpath("trained_calls.json"))
    write_json(
        {"input_shape": input_shape, "num_labels": num_labels},
        file_paths["model_dir"].joinpath("shape.json"),
    )
    # TODO: Save model?
    model.save(file_paths["model"])

    msgr.success(
        f"OrcAI - training model finished. Model saved to {file_paths['model']}"
    )
    return
