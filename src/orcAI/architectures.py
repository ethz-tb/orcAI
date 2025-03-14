import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable

from orcAI.auxiliary import Messenger

# TODO: docstrings for functions, try pyment


# CNN model with residual connection
def res_net_1Dconv_arch(
    input_shape: tuple[int, int, int],
    num_labels: int,
    filters: list[int],
    kernel_size: int,
    dropout_rate: float,
    **unused,
):
    """TensorFlow/Keras model architecture for a Convolutional Neural Network
    (CNN) with residual connections (ResNet) followed by a global temporal
    aggregation step using a 1D convolution

    Parameters
    ----------
    input_shape : tuple (int, int, int)
        Dimensions of the input data
    num_labels : int
        Number of labels to predict
    filters : list of int
        Number of filters in each convolutional layer
    kernel_size : int
        Size of the convolutional kernel
    dropout_rate : float
        Dropout rate for the model
    **unused :
        Additional keyword arguments, unused


    Returns
    -------
    tf.keras.Model
        Model architecture

    """
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


def res_net_LSTM_arch(
    input_shape: tuple[int, int, int],
    num_labels: int,
    filters: list[int],
    kernel_size: int,
    dropout_rate: float,
    lstm_units: int,
    **unused,
):
    """TensorFlow/Keras model architecture for a Convolutional Neural Network
    (CNN) with residual connections (ResNet) extended with bidirectional
    Long Short-Term Memory (LSTM) layers

    Parameters
    ----------
    input_shape : tuple (int, int, int)
        Dimensions of the input data
    num_labels : int
        Number of labels to predict
    filters : list of int
        Number of filters in each convolutional layer
    kernel_size : int
        Size of the convolutional kernel
    dropout_rate : float
        Dropout rate for the model
    lstm_units : int
        Dimensionality of the output space of the LSTM layer
    **unused :
        Additional keyword arguments, unused

    Returns
    -------
    tf.keras.Model
        Model architecture

    """
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


# define masked binary crossentropy and masked binary accuracy
@register_keras_serializable(name="masked_binary_crossentropy")
def masked_binary_crossentropy(y_true: any, y_pred: any):
    """Custom binary cross-entropy loss function with label masking.

    Parameters
    ----------
    y_true : any
        True labels (with -1 indicating missing labels).
    y_pred : any
        Predicted probabilities for each label.

    Returns
    -------
    tf.reduce_mean(loss) :
        The reduced tensor

    """
    mask_value = -1.0
    # Ensure mask_value has the same type as y_true
    mask_value = tf.cast(mask_value, y_true.dtype)

    # Create a mask: where y_true != mask_value
    mask = tf.not_equal(y_true, mask_value)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Standard binary cross-entropy on the masked values
    loss = tf.keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)
    return tf.reduce_mean(loss)


@register_keras_serializable(name="masked_binary_accuracy")
def masked_binary_accuracy(y_true: any, y_pred: any):
    """Custom binary accuracy metric that excludes masked labels.

    Parameters
    ----------
    y_true : any
        True labels (with -1 indicating missing labels).
    y_pred : any
        Predicted probabilities.

    Returns
    -------
    tf.reduce_mean(accuracy) :
        The reduced tensor


    """
    mask_value = -1.0
    # Ensure mask_value has the same type as y_true
    mask_value = tf.cast(mask_value, y_true.dtype)
    # Create a mask: where y_true != mask_value
    mask = tf.not_equal(y_true, mask_value)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Compute binary accuracy on masked values
    accuracy = tf.keras.metrics.binary_accuracy(y_true_masked, y_pred_masked)
    return tf.reduce_mean(accuracy)


@register_keras_serializable(name="masked_f1_score")
def masked_f1_score(y_true: any, y_pred: any, threshold: float = 0.5):
    """Custom F1 metric that excludes masked labels.

    Parameters
    ----------
    y_true : any
        True labels (with -1 indicating missing labels).
    y_pred : any
        Predicted probabilities or logits.
    threshold : float
        Threshold above which predictions are considered 1, else 0. (Default value = 0.5)

    Returns
    -------
    f1 : float
        Scalar F1 score (float) for the unmasked elements in this batch.

    """
    mask_value = -1.0
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


ORCAI_ARCHITECTURES_FN = {
    "ResNet1DConv": res_net_1Dconv_arch,
    "ResNetLSTM": res_net_LSTM_arch,
}

ORCAI_ARCHITECTURES = list(ORCAI_ARCHITECTURES_FN.keys())


# build model from a choice of models
def build_model(
    input_shape: tuple[int, int, int],
    num_labels: int,
    model_parameter: dict,
    msgr: Messenger = Messenger(),
):
    """

    Parameters
    ----------
    input_shape : tuple (int, int, int)
        Dimensions of the input data
    num_labels : int
        Number of labels to predict
    model_parameter : dict
        Model parameter dictionary
    msgr : Messenger
        Messenger object for messages
         (Default value = Messenger())

    Returns
    -------
        model : tf.keras.Model
            Model
    """

    if model_parameter["architecture"] in ORCAI_ARCHITECTURES:
        model = ORCAI_ARCHITECTURES_FN[model_parameter["architecture"]](
            input_shape, num_labels, **model_parameter
        )
    else:
        raise ValueError(
            f"Unknown model architecture: {model_parameter['architecture']}"
        )

    n_filters = len(model_parameter["filters"])
    output_shape = (input_shape[0] // 2**n_filters, num_labels)

    msgr.part("Building model architecture")
    msgr.info(f"model name:          {model_parameter['name']}")
    msgr.info(f"model architecture:  {model_parameter['architecture']}")
    msgr.info(f"model input shape:   {model.input_shape}")
    msgr.info(f"model output shape:  {model.output_shape}")
    msgr.info(f"actual input_shape:  {input_shape}")
    msgr.info(f"actual output_shape: {output_shape}")
    msgr.info(f"n_filters:           {n_filters}")
    msgr.info(f"num_labels:          {num_labels}")
    return model
