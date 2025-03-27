from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.saving import load_model

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)

from orcAI.auxiliary import (
    Messenger,
    SEED_ID_LOAD_TEST_DATA,
    SEED_ID_UNFILTERED_TEST_DATA,
)

from orcAI.architectures import (
    build_model,
    masked_binary_accuracy,
    masked_binary_crossentropy,
)
from orcAI.io import DataLoader, load_dataset, read_json


def _stack_batch(batch):
    """Stack a batch of label matrices."""
    stacked = np.vstack(batch).astype(int)
    return stacked


def _get_mask_for_rows_with_atmost_one_1(matrix):
    """Get a boolean mask for rows with at most one '1' in a matrix."""
    count_ones_per_row = np.sum(matrix == 1, axis=1)
    # Create a boolean mask: True if the row has <= 1 '1'
    mask = count_ones_per_row <= 1
    return mask


def _compute_misclassification_table(
    label_matrix_1, label_matrix_2, suffix_1, suffix_2, label_names
):
    """Compute the misclassification table between two label matrices.

    Parameters
    ----------
    label_matrix_1: np.ndarray
        Binary label matrix.
    label_matrix_2: np.ndarray
        Binary label matrix.
    suffix_1: str
        Suffix for the first label matrix. (e.g. "true")
    suffix_2: str
        Suffix for the second label matrix. (e.g. "pred")
    label_names: list[str]
        List of label names.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the misclassification table (1 vs 2).
    """
    num_labels = len(label_names)
    misclassification_matrix = np.zeros((num_labels + 1, num_labels + 1))

    for row_index in range(label_matrix_1.shape[0]):
        col_mat1_equal_one = np.where(label_matrix_1[row_index, :] == 1)[0]
        cols_mat2_equal_one = np.where(label_matrix_2[row_index, :] == 1)[0]
        if len(col_mat1_equal_one) == 1:
            if (
                label_matrix_2[row_index, col_mat1_equal_one] != -1
            ):  # only proceed if column in second matrix is not masked
                if len(cols_mat2_equal_one) > 0:
                    for cp_i in cols_mat2_equal_one:
                        misclassification_matrix[col_mat1_equal_one, cp_i] += 1 / len(
                            cols_mat2_equal_one
                        )
                else:
                    misclassification_matrix[col_mat1_equal_one, num_labels] += 1
        if len(col_mat1_equal_one) == 0:
            if len(cols_mat2_equal_one) > 0:
                for cp_i in cols_mat2_equal_one:
                    misclassification_matrix[num_labels, cp_i] += 1 / len(
                        cols_mat2_equal_one
                    )
            else:
                misclassification_matrix[num_labels, num_labels] += 1
        if len(col_mat1_equal_one) > 1:
            print("WARNING: more than one 1 in row of matrix y_true_stacked_drop")

    # normalize confusion matrix
    row_sum = np.sum(misclassification_matrix, axis=1, keepdims=True)
    misclassification_matrix = misclassification_matrix / row_sum
    misclassification_matrix = np.around(misclassification_matrix, 3)
    col_names = [suffix_2 + "_" + x for x in label_names]
    col_names += [suffix_2 + "_NOLABEL"]
    row_names = [suffix_1 + "_" + x for x in label_names]
    row_names += [suffix_1 + "_NOLABEL"]
    misclassification_table = pd.DataFrame(misclassification_matrix)
    misclassification_table.columns = col_names
    misclassification_table.index = row_names
    misclassification_table["fraction_time"] = np.around(row_sum / sum(row_sum), 5)

    return misclassification_table


def compute_misclassification_tables(
    label_matrix_1, label_matrix_2, suffix_1, suffix_2, label_names
):
    """Compute both misclassification tables for two label matrices (predicted and true).

    Parameters
    ----------
    label_matrix_1: np.ndarray
        Binary label matrix.
    label_matrix_2: np.ndarray
        Binary label matrix.
    suffix_1: str
        Suffix for the first label matrix. (e.g. "true")
    suffix_2: str
        Suffix for the second label matrix. (e.g. "pred")
    label_names: list[str]
        List of label names.

    Returns
    -------
    dict
        A dictionary containing both misclassification tables (e.g. "true_pred" and "pred_true").


    """
    label_matrix_1_mask = _get_mask_for_rows_with_atmost_one_1(label_matrix_1)
    label_matrix_2_mask = _get_mask_for_rows_with_atmost_one_1(label_matrix_2)

    misclassification_table_1_2 = _compute_misclassification_table(
        label_matrix_1[label_matrix_1_mask],
        label_matrix_2[label_matrix_1_mask],
        suffix_1,
        suffix_2,
        label_names,
    )
    misclassification_table_2_1 = _compute_misclassification_table(
        label_matrix_2[label_matrix_2_mask],
        label_matrix_1[label_matrix_2_mask],
        suffix_2,
        suffix_1,
        label_names,
    )
    return {
        "_".join([suffix_1, suffix_2]): misclassification_table_1_2,
        "_".join([suffix_2, suffix_1]): misclassification_table_2_1,
    }


def compute_confusion_table(
    y_true_batch: np.ndarray,
    y_pred_batch: np.ndarray,
    label_names: list[str],
):
    """Compute the confusion matrix for each label across the entire batch.

    Parameters:
    ----------
    y_true_batch: np.ndarray
        Ground truth binary labels with shape (batch_size, time_steps, num_labels).
    y_pred_batch: np.ndarray
        Predicted labels with shape (batch_size, time_steps, num_labels).
    label_names: list[str]
        List of label names.

    Returns
    -------
    confusion_table: pd.DataFrame
        A DataFrame with confusion values for each label.

    """
    mask_value = -1
    # Ensure inputs are numpy arrays
    y_true_batch = np.array(y_true_batch)
    y_pred_binary_batch = (y_pred_batch >= 0.5).astype(int)
    y_pred_binary_batch = np.array(y_pred_binary_batch)

    # Validate input shapes
    assert (
        y_true_batch.shape == y_pred_binary_batch.shape
    ), "Shapes of y_true_batch and y_pred_binary_batch must match"

    # Initialize a dictionary to store confusion matrices for each label
    confusion_table = {}

    for label_idx in range(len(label_names)):
        # Flatten the predictions and ground truth for the current label
        y_true_flat = y_true_batch[:, :, label_idx].flatten()
        y_pred_flat = y_pred_binary_batch[:, :, label_idx].flatten()

        # Apply the mask to exclude masked values
        mask = y_true_flat != mask_value
        y_true_filtered = y_true_flat[mask]
        y_pred_filtered = y_pred_flat[mask]

        # Compute the confusion matrix for the current label
        [tn, fp], [fn, tp] = confusion_matrix(
            y_true_filtered, y_pred_filtered, labels=[0, 1]
        )
        tot = tn + fp + fn + tp
        cm = {
            "TP": float(tp / tot),
            "FN": float(fn / tot),
            "FP": float(fp / tot),
            "TN": float(tn / tot),
            "PR": float(tp / (tp + fp)) if tp + fp > 0 else np.nan,
            "RE": float(tp / (tp + fn)) if tp + fn > 0 else np.nan,
            "F1": float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn > 0 else np.nan,
            "Total": int(tot),
        }
        # Store the confusion matrix
        confusion_table[label_names[label_idx]] = cm
    confusion_table = pd.DataFrame.from_dict(
        confusion_table, orient="index"
    ).sort_values(by="Total", ascending=False)
    return confusion_table


def _test_model_on_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    label_names: list[str],
    dataset_name: str,
    dataset_length: int,
    msgr: Messenger,
):
    """Test a model on a dataset."""
    msgr.part(f"Testing model on {dataset_name}", indent=1)
    msgr.info(f"Evaluating model on {dataset_name}")
    data_loss, data_metric = model.evaluate(
        dataset, verbose=0 if msgr.verbosity < 3 else 1
    )
    msgr.info(f"loss: {data_loss:.4f}")
    msgr.info(f"masked binary accuracy: {data_metric:.4f}")
    msgr.debug(f"Input (spectrogram) shape: {model.input_shape}")
    msgr.debug(f"Output (labels) shape: {model.output_shape}", indent=-1)

    # CONFUSION MATRICES
    msgr.part(f"Calculating confusion table for {dataset_name}")
    data_predicted = []
    data_true = []

    for spectrogram_batch, label_batch in tqdm(
        dataset,
        disable=True if msgr.verbosity < 2 else None,
        desc="predicting data",
        total=dataset_length,
    ):
        data_true.append(label_batch.numpy())
        data_predicted.append(model.predict(spectrogram_batch, verbose=0))

    data_true = np.concatenate(data_true, axis=0)
    data_predicted = np.concatenate(data_predicted, axis=0)

    confusion_table = compute_confusion_table(data_true, data_predicted, label_names)
    msgr.info(confusion_table)

    # MISCLASSIFICATION TABLES on dataset
    data_true_stacked = _stack_batch(data_true)
    data_predicted_stacked = _stack_batch((data_predicted >= 0.5).astype(int))

    # compute misclassification
    missclassification_tables = compute_misclassification_tables(
        label_matrix_1=data_true_stacked,
        label_matrix_2=data_predicted_stacked,
        suffix_1="true",
        suffix_2="pred",
        label_names=label_names,
    )
    msgr.part("Misclassification tables on dataset:")
    for key, table in missclassification_tables.items():
        msgr.info("\n" + key, indent=1)
        msgr.info(table, indent=-1)

    return {
        "dataset": dataset_name,
        "loss": data_loss,
        "masked_binary_accuracy": data_metric,
        "confusion_table": confusion_table,
        "misclassification_tables": missclassification_tables,
    }


def _save_test_results(
    results: dict,
    save_results_dir: Path,
    msgr: Messenger,
):
    """Save test results to disk."""
    msgr.part(f"Saving test results")
    dataset_name = results["dataset"]

    metrics = {
        key: value
        for key, value in results.items()
        if key in ["loss", "masked_binary_accuracy"]
    }
    with open(save_results_dir.joinpath(dataset_name + "_metrics.json"), "w") as f:
        json.dump(metrics, f)

    results["confusion_table"].to_csv(
        save_results_dir.joinpath(dataset_name + "_confusion_table.csv")
    )

    for key, table in results["misclassification_tables"].items():
        table.to_csv(
            save_results_dir.joinpath(
                dataset_name + "_" + "misclassification_table_" + key + ".csv"
            )
        )
    return


def test_model(
    model_dir: Path | str,
    data_dir: Path | str,
    recording_data_dir: Path | str | None = None,
    n_batches_additional: int = 3200,
    output_dir: None | Path | str = None,
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> None:
    """Test a trained model on test data and a sample of test snippets."
    Parameters
    ----------
    model_dir : Path | str
        Path to the model directory.
    data_dir : Path | str
        Path to the model data directory containing the training, valdidation
        and testing data.
    recording_data_dir : (Path | str)
        Path to the recording data directory
    test_data_sample_size : int
        Number of test snippets to sample from the complete test data for testing. Default is 100000.
    output_dir : None | Path | str
        Directory to save the test results. Default is saving to model_dir.
    verbosity : int
        Verbosity level. 0: Errors only, 1: Warnings, 2: Info, 3: Debug
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.

    Returns
    -------
    None
        Saves results to disk if save_results_dir is provided.
    """

    # Initialize messenger
    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Testing model")
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir
    else:
        output_dir = Path(output_dir)

    msgr.part("Loading parameter and data")
    msgr.info(f"Model directory: {model_dir}")
    msgr.info(f"Model data directory: {data_dir}")

    orcai_parameter = read_json(Path(model_dir).joinpath("orcai_parameter.json"))
    model_parameter = orcai_parameter["model"]
    msgr.debug("Model parameter")
    msgr.debug(model_parameter)

    trained_calls = orcai_parameter["calls"]

    # LOAD MODEL #TODO: load from .keras file?
    msgr.part("Loading model")

    model = load_model(
        model_dir.joinpath(orcai_parameter["name"] + ".keras"),
        custom_objects=None,
        compile=True,
        safe_mode=True,
    )

    msgr.part("Testing model on test data")
    dataset_shape = read_json(data_dir.joinpath("dataset_shapes.json"))
    test_dataset = load_dataset(
        data_dir.joinpath("test_dataset.tfrecord.gz"),
        dataset_shape,
        model_parameter["batch_size"],
        [
            SEED_ID_LOAD_TEST_DATA,
            orcai_parameter["seed"],
        ],
    )
    results_test_dataset = _test_model_on_dataset(
        model,
        test_dataset,
        trained_calls,
        "test_data",
        model_parameter["n_batch_test"],
        msgr,
    )
    _save_test_results(results_test_dataset, output_dir, msgr)
    msgr.info(f"Saved test results to {output_dir}")

    if recording_data_dir is not None:
        msgr.part("Testing model on new, unfiltered sample of test snippets")
        all_snippets = pd.read_csv(recording_data_dir.joinpath("all_snippets.csv.gz"))
        all_test_snippets = all_snippets[all_snippets["data_type"] == "test"]
        test_data_sample_size = n_batches_additional * model_parameter["batch_size"]
        rng = np.random.default_rng(
            seed=[SEED_ID_UNFILTERED_TEST_DATA, orcai_parameter["seed"]]
        )
        if len(all_test_snippets) < test_data_sample_size:
            msgr.warning(
                f"Test data sample size ({test_data_sample_size}) is larger than the number of test snippets ({len(all_test_snippets)})."
            )
            msgr.warning("Using all test snippets for testing.")
            test_data_sample_size = len(all_test_snippets)

        sampled_test_snippets = all_test_snippets.sample(
            test_data_sample_size, replace=False, random_state=rng
        ).reset_index()
        sampled_test_snippets_loader = DataLoader(
            sampled_test_snippets,
            n_filters=len(model_parameter["filters"]),
        )
        test_sampled_dataset = tf.data.Dataset.from_generator(
            sampled_test_snippets_loader.__iter__,
            output_signature=(
                tf.TensorSpec(
                    shape=(model.input_shape[1], model.input_shape[2], 1),
                    dtype=tf.float32,
                ),  # Single spectrogram shape
                tf.TensorSpec(
                    shape=(model.output_shape[1], model.output_shape[2]),
                    dtype=tf.float32,
                ),  # Single label shape
            ),
        )
        test_sampled_dataset = test_sampled_dataset.batch(
            model_parameter["batch_size"], drop_remainder=True
        ).prefetch(buffer_size=tf.data.AUTOTUNE)
        total_batches = (
            len(sampled_test_snippets_loader) // model_parameter["batch_size"]
        )
        test_sampled_dataset = test_sampled_dataset.apply(
            tf.data.experimental.assert_cardinality(total_batches)
        )
        results_test_dataset = _test_model_on_dataset(
            model,
            test_sampled_dataset,
            trained_calls,
            "test_sampled_data",
            total_batches,
            msgr,
        )
        _save_test_results(results_test_dataset, output_dir, msgr)
        msgr.info(f"Saved test results to {output_dir}")

    msgr.success("Model testing completed.")

    return
