import tensorflow as tf
from tensorflow.keras import layers

from orcAI.auxiliary import Messenger

# TODO: docstrings for functions


# CNN model with residual connection (corresponds to old model)
def build_cnn_res_arch(
    input_shape, num_labels, filters, kernel_size, dropout_rate, **unused
):
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(16, kernel_size, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.Dropout(0.3)(x)  # Dropout after the first layer

    previous_block_activation = x  # Set aside residual

    for size in filters:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 2), strides=(2, 2), padding="same")(x)
        residual = layers.Conv2D(size, 1, strides=(2, 2), padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        x = layers.Dropout(dropout_rate)(x)  # Dropout

    x = layers.SeparableConv2D(36, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout after the final CNN block

    x = tf.reduce_mean(x, axis=2)
    # 1D convolutional layer over time axis
    k_size = x.shape[2]
    outputs = layers.Conv1D(
        num_labels, kernel_size=k_size, padding="same", activation="sigmoid"
    )(x)

    return tf.keras.Model(inputs, outputs)


# CNN RES LSTM Model
def build_cnn_res_lstm_arch(
    input_shape, num_labels, filters, kernel_size, dropout_rate, lstm_units, **unused
):
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(16, kernel_size, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Residual blocks
    for size in filters:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 2), strides=(2, 2), padding="same")(x)
        residual = layers.Conv2D(size, 1, strides=(2, 2), padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Final CNN processing block
    x = layers.SeparableConv2D(36, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Reshape for LSTM input
    x = layers.Reshape(target_shape=(-1, x.shape[-2] * x.shape[-1]))(x)

    # LSTM layers with regularization
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # Fully connected layers with regularization
    x = layers.Dense(
        128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_labels, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


# cnn_res_transformer model
def build_cnn_res_transformer_arch(
    input_shape, num_labels, filters, kernel_size, dropout_rate, num_heads, **unused
):
    inputs = tf.keras.Input(shape=input_shape)

    # CNN layers
    x = layers.Conv2D(16, kernel_size, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x

    for size in filters:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 2), strides=(2, 2), padding="same")(x)
        residual = layers.Conv2D(size, 1, strides=(2, 2), padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    # Final CNN block
    x = layers.SeparableConv2D(36, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Reshape and add positional encodings
    x = layers.Reshape(target_shape=(-1, x.shape[-2] * x.shape[-1]))(x)
    seq_len = tf.shape(x)[1]
    embedding_dim = 396

    # Positional encodings with matching dimensions
    positional_encodings = layers.Lambda(
        lambda inputs: tf.range(start=0, limit=inputs[0], delta=1)
    )([seq_len])
    positional_encodings = layers.Embedding(
        input_dim=50,  # Upper limit for sequence length
        output_dim=embedding_dim,  # Match transformer_units to x's embedding dimension
    )(positional_encodings)
    x += positional_encodings

    # Transformer layers
    for _ in range(2):
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim
        )(x, x)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        feed_forward = layers.Dense(embedding_dim, activation="relu")(x)
        feed_forward = layers.Dense(embedding_dim)(feed_forward)
        feed_forward = layers.Dropout(dropout_rate)(feed_forward)
        x = layers.Add()([x, feed_forward])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Output layers
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_labels, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


# TODO: which function is the correct one?
# cnn_res_transformer model
def build_cnn_res_transformer_arch_new(
    input_shape, num_labels, filters, kernel_size, dropout_rate, num_heads, **unused
):
    inputs = tf.keras.Input(shape=input_shape)

    # CNN layers
    x = layers.Conv2D(16, kernel_size, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x

    for size in filters:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 2), strides=(2, 2), padding="same")(x)
        residual = layers.Conv2D(size, 1, strides=(2, 2), padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    # Final CNN block
    x = layers.SeparableConv2D(36, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Reshape and add positional encodings
    x = layers.Reshape(target_shape=(-1, x.shape[-2] * x.shape[-1]))(x)
    seq_len = tf.shape(x)[1]
    embedding_dim = 396

    # Positional encodings with matching dimensions
    positional_encodings = layers.Lambda(
        lambda inputs: tf.range(start=0, limit=inputs[0], delta=1)
    )([seq_len])
    positional_encodings = layers.Embedding(
        input_dim=50,  # Upper limit for sequence length
        output_dim=embedding_dim,  # Match transformer_units to x's embedding dimension
    )(positional_encodings)
    x += positional_encodings

    # Transformer layers
    for _ in range(2):
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim
        )(x, x)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        feed_forward = layers.Dense(embedding_dim, activation="relu")(x)
        feed_forward = layers.Dense(embedding_dim)(feed_forward)
        feed_forward = layers.Dropout(dropout_rate)(feed_forward)
        x = layers.Add()([x, feed_forward])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Output layers
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_labels, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


# define masked binary crossentropy and masked binary accuracy
def masked_binary_crossentropy(y_true, y_pred, mask_value=-1.0):
    """
    Custom binary cross-entropy loss function with label masking.

    Args:
        y_true: True labels (with -1 or a mask_value indicating missing labels).
        y_pred: Predicted probabilities for each label.
        mask_value: Value used to mask missing labels.

    Returns:
        Loss scalar.
    """
    # Ensure mask_value has the same type as y_true
    mask_value = tf.cast(mask_value, y_true.dtype)

    # Create a mask: where y_true != mask_value
    mask = tf.not_equal(y_true, mask_value)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Standard binary cross-entropy on the masked values
    loss = tf.keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)
    return tf.reduce_mean(loss)


def masked_binary_accuracy(y_true, y_pred, mask_value=-1.0):
    """
    Custom binary accuracy metric that excludes masked labels.

    Args:
        y_true: True labels (with -1 or mask_value indicating missing labels).
        y_pred: Predicted probabilities.
        mask_value: Value used to mask missing labels.

    Returns:
        Masked accuracy.
    """

    # Ensure mask_value has the same type as y_true
    mask_value = tf.cast(mask_value, y_true.dtype)
    # Create a mask: where y_true != mask_value
    mask = tf.not_equal(y_true, mask_value)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Compute binary accuracy on masked values
    accuracy = tf.keras.metrics.binary_accuracy(y_true_masked, y_pred_masked)
    return tf.reduce_mean(accuracy)


def masked_f1_score(y_true, y_pred, mask_value=-1.0, threshold=0.5):
    """
    Custom F1 metric that excludes masked labels.

    Args:
        y_true: True labels (with -1 or mask_value indicating missing labels).
        y_pred: Predicted probabilities or logits.
        mask_value: Value used to mask missing labels.
        threshold: Threshold above which predictions are considered 1, else 0.

    Returns:
        Scalar F1 score (float) for the unmasked elements in this batch.
    """
    # Ensure y_true is float
    y_true = tf.cast(y_true, tf.float32)
    mask_value = tf.cast(mask_value, tf.float32)

    # Create a mask for valid (non-masked) elements
    mask = tf.not_equal(y_true, mask_value)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Binarize the predictions
    y_pred_bin = tf.cast(y_pred_masked >= threshold, tf.float32)

    # Calculate confusion matrix components
    tp = tf.reduce_sum(y_true_masked * y_pred_bin)  # 1 & 1
    fp = tf.reduce_sum((1 - y_true_masked) * y_pred_bin)  # 0 & 1
    fn = tf.reduce_sum(y_true_masked * (1 - y_pred_bin))  # 1 & 0

    # Avoid division by zero
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1


def reshape_labels(arr, n_filters):
    """
    Reshape and process labels using the provided number of filters (n_filters) to achieve a time resolution on labels which is time_steps_spectogram//2**n_filters.
    """

    if arr.shape[0] % (2**n_filters) == 0:
        # Reshape the array to group rows for averaging
        new_shape = (
            arr.shape[0] // (2**n_filters),
            2**n_filters,
            arr.shape[1],
        )
        reshaped = tf.reshape(
            arr, new_shape
        )  # Shape: (time_steps_labels, downsample_factor, num_labels)
        # Compute the mean along the downsampling axis
        averaged = tf.reduce_mean(
            reshaped, axis=1
        )  # Shape: (time_steps_labels, num_labels)
        arr_out = tf.round(averaged)  # round to next integer
        return tf.cast(arr_out, dtype=tf.int32)
    else:
        raise ValueError(
            "The number of rows in 'arr' must be divisible by 2**'n_filters'."
        )


ORCAI_METRICS_FN = {
    "masked_binary_accuracy": masked_binary_accuracy,
    "masked_f1_score": masked_f1_score,
}

ORCAI_METRICS = list(ORCAI_METRICS_FN.keys())


def choose_metric(metric_name):
    if metric_name in ORCAI_METRICS:
        return ORCAI_METRICS_FN[metric_name]
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")


ORCAI_ARCHITECTURES_FN = {
    "cnn_res_model": build_cnn_res_arch,
    "cnn_res_lstm_model": build_cnn_res_lstm_arch,
    "cnn_res_transformer_model": build_cnn_res_transformer_arch,
    "cnn_res_transformer_model_new": build_cnn_res_transformer_arch_new,
}

ORCAI_ARCHITECTURES = list(ORCAI_ARCHITECTURES_FN.keys())


# build model from a choice of models
def build_model(input_shape, num_labels, model_dict, msgr=Messenger()):
    n_filters = len(model_dict["filters"])
    output_shape = (input_shape[0] // 2**n_filters, num_labels)

    if model_dict["name"] in ORCAI_ARCHITECTURES:
        model = ORCAI_ARCHITECTURES_FN[model_dict["name"]](
            input_shape, num_labels, **model_dict
        )
    else:
        raise ValueError(f"Unknown model name: {model_dict['name']}")

    msgr.part("Building model architecture")
    msgr.info(f"model name:          {model_dict['name']}")
    msgr.info(f"model input shape:   {model.input_shape}")
    msgr.info(f"model output shape:  {model.output_shape}")
    msgr.info(f"actual input_shape:  {input_shape}")
    msgr.info(f"actual output_shape: {output_shape}")
    msgr.info(f"n_filters:           {n_filters}")
    msgr.info(f"num_labels:          {num_labels}")
    return model
