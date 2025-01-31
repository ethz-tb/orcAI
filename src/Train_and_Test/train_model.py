import time
import numpy as np
import tensorflow as tf
from os import path
from importlib.resources import files
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.backend import count_params

# import local
import auxiliary as aux
import model as mod
import load

# model parameters
def count_params(trainable_weights):
    return np.sum([np.prod(w.shape) for w in trainable_weights])

def train_model(model_name, data_dir, project_dir, load_weights, calls_for_labeling, verbosity):
    """Trains an orcAI model
    
    expects the following file structure:
        /(data_dir)/
            val_dataset
            test_dataset
            train_dataset
        /(project_dir)/
            Results/
                (model_name)/
                    model.dict
                    (model_name) # for model weigths
        # GenericParameters/
        #     calls_for_labeling.list
    """
    print("OrcAI - training model ...")
    print("Project directory:", project_dir)

    print("READ IN PARAMETERS")

    file_paths = {
        "model_path": path.join(project_dir, "Results", model_name, model_name),
        "model_parameter": path.join(project_dir, "Results", model_name, "model.dict"),
        "calls_labeling": files('data').joinpath('calls_for_labeling.json') if calls_for_labeling=="default" else calls_for_labeling,
        "training_data": path.join(data_dir, "train_dataset"),
        "validation_data": path.join(data_dir, "val_dataset"),
        "test_data": path.join(data_dir, "test_dataset"),
        "history": path.join(project_dir, "Results", model_name, "training_history.json"),
        "confusion_matrices": path.join(project_dir , "Results", model_name, "confusion_matrices.json")
    }

    print("  - reading model parameters")
    model_dict = aux.read_dict(file_paths["model_parameter"], True)
    print("  - reading calls for labeling")
    calls_for_labeling_list = aux.read_dict(file_paths["calls_labeling"], True)

    # this complicates static code analysis
    #
    # dicts = {
    #     "model_dict": path.join(project_dir, "Results", model_name, "/model.dict"),
    #     "calls_for_labeling_list": path.join("GenericParameters/calls_for_labeling.list"),
    # }
    # for key, value in dicts.items():
    #     print("  - reading", key)
    #     globals()[key] = aux.read_dict(value, True)

    # load data sets from local disk
    print("Loading train, val and test datasets from disk:", data_dir)
    start_time = time.time()
    train_dataset = load.reload_dataset(file_paths["training_data"], model_dict["batch_size"])
    val_dataset = load.reload_dataset(file_paths["validation_data"], model_dict["batch_size"])
    test_dataset = load.reload_dataset(file_paths["test_data"], model_dict["batch_size"])
    print(f"  - time to load datasets: {time.time() - start_time:.2f} seconds")

    # Verify the val dataset and obtain shape
    for spectrogram, labels in val_dataset.take(1):
        print("Spectrogram batch shape:", spectrogram.numpy().shape)
        print("Labels batch shape:", labels.numpy().shape)

    # Build model
    input_shape = tuple(spectrogram.shape[1:])  #  shape
    num_labels = labels.shape[2]  # Number of sound types

    model = mod.build_model(input_shape, num_labels, model_dict)

    #TRANSFORMER MODEL FIX
    transformer_parallel = False
    if transformer_parallel:
        if model_name == "cnn_res_transformer_model":
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
                    **model_dict
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
    print("Fitting mode:", model_name)
    if load_weights:
        print("  - Loading weights from stored model:", model_name)
        model.load_weights(file_paths["model_path"])
    else:
        print("  - Learning weights from scratch")

    # Compiling Model
    print("Compiling model:", model_name)
    # Metric
    masked_binary_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(
        fn=lambda y_true, y_pred: mod.masked_binary_accuracy(
            y_true, y_pred, mask_value=-1.0
        ),
        name="masked_binary_accuracy",
    )
    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_masked_binary_accuracy",  # Use the validation metric
        patience=model_dict["patience"],  # Number of epochs to wait for improvement
        mode="max",  # Stop when accuracy stops increasing
        restore_best_weights=True,  # Restore weights from the best epoch
    )
    model_checkpoint = ModelCheckpoint(
        file_paths["model_path"],
        monitor="val_masked_binary_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_masked_binary_accuracy",  # Monitor your custom validation metric
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
        metrics=[masked_binary_accuracy_metric],
    )

    total_params = model.count_params()
    trainable_params = count_params(model.trainable_weights)
    non_trainable_params = count_params(model.non_trainable_weights)
    print("Model size:")
    print(f"  - Total parameters: {total_params}")
    print(f"  - Trainable parameters: {trainable_params}")
    print(f"  - Non-trainable parameters: {non_trainable_params}")
    aux.print_memory_usage()

    # Train model
    print("Training model:", model_name)
    start_time = time.time()

    with tf.device("/GPU:0"):
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=model_dict["epochs"],
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=verbosity,
        )
    print(f"  - total time for training: {time.time() - start_time:.2f} seconds")

    print("  - training history dictionary:", history.history)
    print("  - saving training history:", file_paths["history"])
    with open(file_paths["history"], "w") as f:
        f.write(str(history.history))

    # Model evaluation
    print("Evaluate model:", model_name)
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
        file_paths["confusion_matrices"],
    )
    
    print("PROGRAM COMPLETED")
