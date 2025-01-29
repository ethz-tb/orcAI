# %%
# import
import tensorflow as tf
from tensorflow.keras import layers, models, Model


# build model from a choice of models


def build_model(model_choice_dict, model_dict, input_shape, num_labels):
    n_filters = len(model_dict["filters"])
    output_shape = (input_shape[0] // 2**n_filters, num_labels)
    if model_dict["name"] in model_choice_dict:
        model = model_choice_dict[model_dict["name"]]()
    else:
        raise ValueError(f"Unknown model name: {model_dict['name']}")
    print("  - model name:", model_dict["name"])
    print("  - model input shape:", model.input_shape)
    print("  - model output shape:", model.output_shape)
    print("  - actual input_shape:", input_shape)
    print("  - actual output_shape:", output_shape)
    print("  - n_filters:", n_filters)
    print("  - num_labels:", num_labels)
    return model


# CNN model with residual connection (corresponds to old mode)


def build_cnn_res_model(input_shape, num_labels, filters, kernel_size, dropout_rate):
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


def build_cnn_lstm_model(input_shape, num_labels):
    """
    Build a CNN-LSTM model for multi-label spectrogram classification with 16-fold downsampling.

    Args:
        input_shape (tuple): Shape of the input spectrogram (time_steps, freq_bins, 1).
        num_labels (int): Number of sound types to classify.

    Returns:
        tf.keras.Model: Compiled model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # 2D CNN layers for spatial feature extraction
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # Downsample by 2
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Downsample by 2

    # Flatten the frequency axis to prepare for temporal processing
    x = layers.Reshape(target_shape=(-1, x.shape[-2] * x.shape[-1]))(
        x
    )  # Shape: (time_steps_spectrogram//4, freq_bins*channels)

    # Adjust the time resolution to match 46 time steps
    x = layers.MaxPooling1D(pool_size=4)(x)  # Downsample by 4 (184 -> 46)

    # Bidirectional GRU instead of LSTM for temporal dependency learning
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)

    # Dense layer for multi-label classification with dropout
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_labels, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model


# CNN RES LSTM Model


def build_cnn_res_lstm_model(
    input_shape, num_labels, filters, kernel_size, dropout_rate, lstm_units
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
def build_cnn_res_transformer_model(
    input_shape,
    num_labels,
    filters,
    kernel_size,
    dropout_rate,
    num_heads,
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
        input_dim=1000,  # Upper limit for sequence length
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


# %%
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


### OLD / EXPERIMENTAL MODELS
# Function to create model
def build_orig_model(input_shape, num_classes, filters):
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in filters:
        # Test: smaller size, remove first batchnorm, MAxPooling to Average pooling
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((3, 2), strides=(2, 2), padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=(2, 2), padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(36, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = tf.reduce_mean(x, axis=2)

    # 1D convolutional layer over time axis
    outputs = layers.Conv1D(
        num_classes, kernel_size=10, padding="same", activation="sigmoid"
    )(x)

    return tf.keras.Model(inputs, outputs)


def build_cnn_transformer_model(
    input_shape,
    num_labels,
    n_filters,
    num_heads=4,
    ff_dim=128,
    num_transformer_blocks=2,
):
    """
    Build a CNN-Transformer model for multi-label spectrogram classification with adjustable time resolution.

    Args:
        input_shape (tuple): Shape of the input spectrogram (time_steps, freq_bins, 1).
        num_labels (int): Number of sound types to classify.
        n_filters (int): Determines the downsampling factor (2**n_filters) for time resolution.
        num_heads (int): Number of attention heads in the transformer.
        ff_dim (int): Feed-forward network dimension inside the transformer block.
        num_transformer_blocks (int): Number of transformer blocks.

    Returns:
        tf.keras.Model: Compiled model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # 2D CNN layers for spatial feature extraction
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # Downsample by 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Downsample by 2
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    # Flatten the frequency axis to prepare for temporal processing
    x = layers.Reshape(target_shape=(-1, x.shape[-2] * x.shape[-1]))(
        x
    )  # Shape: (time_steps_spectrogram//4, freq_bins*channels)

    # Adjust the time resolution based on n_filters
    if n_filters == 0:
        # No further downsampling, keep time_steps identical to spectrogram
        pass
    elif n_filters == 1:
        # Downsample the time axis by 2
        x = layers.MaxPooling1D(pool_size=2)(x)
    elif n_filters > 1:
        # Downsample the time axis by 2**(n_filters - 2)
        for _ in range(n_filters - 2):
            x = layers.MaxPooling1D(pool_size=2)(x)

    # Add Transformer blocks for temporal dependency learning
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)

    # Dense layer for multi-label classification
    x = layers.Dense(128, activation="relu")(
        x
    )  # Shape: (batch_size, time_steps_labels, 128)
    outputs = layers.Dense(num_labels, activation="sigmoid")(
        x
    )  # Shape: (batch_size, time_steps_labels, num_labels)

    model = models.Model(inputs, outputs)
    return model


def transformer_block(x, num_heads, ff_dim, dropout_rate=0.1):
    """
    A single transformer block with multi-head attention and feed-forward layers.

    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, time_steps, embedding_dim).
        num_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward network dimension.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tf.Tensor: Transformed tensor of the same shape as input.
    """
    # Multi-Head Self-Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=x.shape[-1]
    )(x, x)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

    # Feed-Forward Network
    ff_output = layers.Dense(ff_dim, activation="relu")(attention_output)
    ff_output = layers.Dense(x.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    ff_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ff_output)

    return ff_output


def build_cnn_res_lstm_model_dropout_L2(input_shape, num_labels, filters):
    """
    Build a CNN-ResNet model with LSTM layers for multi-label classification.

    Args:
        input_shape (tuple): Shape of the input data (time_steps, freq_steps, channels).
        num_labels (int): Number of output labels.
        filters (list): List of filter sizes for the residual blocks.

    Returns:
        tf.keras.Model: Compiled model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(16, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Residual blocks
    for size in filters:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 2), strides=(2, 2), padding="same")(x)
        residual = layers.Conv2D(size, 1, strides=(2, 2), padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Final CNN processing block
    x = layers.SeparableConv2D(36, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Reshape for LSTM input
    x = layers.Reshape(target_shape=(-1, x.shape[-2] * x.shape[-1]))(x)

    # LSTM layers for temporal processing
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # Fully connected layer for classification
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_labels, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)
