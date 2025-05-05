import keras
import tensorflow as tf

from orcAI.auxiliary import MASK_VALUE, Messenger

tf.get_logger().setLevel(40)  # suppress tensorflow logging (ERROR and worse only)


# CNN model with residual connection
class ReduceFrequencyMean(keras.layers.Layer):
    """Custom Keras layer to reduce the frequency dimension of a 4D tensor
    by taking the mean across the frequency axis (axis=2)"""

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=2)


def res_net_1Dconv_arch(
    input_shape: tuple[int, int, int],
    num_labels: int,
    filters: list[int],
    kernel_size: int,
    dropout_rate: float,
    conv_initializer: str = "glorot_uniform",
    **unused,
) -> keras.Model:
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
    conv_initializer : str
        Initializer for the convolutional layers, defaults to "glorot_uniform"
    **unused :
        Additional keyword arguments, unused

    Returns
    -------
    keras.Model
        Model architecture

    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = keras.layers.Conv2D(
        16, kernel_size, padding="same", kernel_initializer=conv_initializer
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x

    for size in filters:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(
            size,
            kernel_size,
            padding="same",
            depthwise_initializer=conv_initializer,
            pointwise_initializer=conv_initializer,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(
            size,
            kernel_size,
            padding="same",
            depthwise_initializer=conv_initializer,
            pointwise_initializer=conv_initializer,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((3, 2), strides=(2, 2), padding="same")(x)
        residual = keras.layers.Conv2D(
            size,
            1,
            strides=(2, 2),
            padding="same",
            kernel_initializer=conv_initializer,
        )(previous_block_activation)
        x = keras.layers.add([x, residual])
        previous_block_activation = x
        x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.SeparableConv2D(
        36,
        kernel_size,
        padding="same",
        depthwise_initializer=conv_initializer,
        pointwise_initializer=conv_initializer,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = ReduceFrequencyMean()(x)
    k_size = x.shape[2]
    outputs = keras.layers.Conv1D(
        num_labels,
        kernel_size=k_size,
        padding="same",
        activation="sigmoid",
        kernel_initializer=conv_initializer,
    )(x)

    return keras.Model(inputs, outputs)


def res_net_LSTM_arch(
    input_shape: tuple[int, int, int],
    num_labels: int,
    filters: list[int],
    kernel_size: int,
    dropout_rate: float,
    lstm_units: int,
    conv_initializer: str = "he_normal",
    lstm_initializer: str = "glorot_uniform",
    **unused,
) -> keras.Model:
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
    conv_initializer : str
        Initializer for the convolutional layers, defaults to "he_normal"
    lstm_initializer : str
        Initializer for the LSTM layers, defaults to "glorot_uniform"
    **unused :
        Additional keyword arguments, unused

    Returns
    -------
    keras.Model
        Model architecture

    """
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(
        16, kernel_size, padding="same", kernel_initializer=conv_initializer
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x

    for size in filters:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(
            size,
            kernel_size,
            padding="same",
            depthwise_initializer=conv_initializer,
            pointwise_initializer=conv_initializer,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(
            size,
            kernel_size,
            padding="same",
            depthwise_initializer=conv_initializer,
            pointwise_initializer=conv_initializer,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((3, 2), strides=(2, 2), padding="same")(x)
        residual = keras.layers.Conv2D(
            size, 1, strides=(2, 2), padding="same", kernel_initializer=conv_initializer
        )(previous_block_activation)
        x = keras.layers.add([x, residual])
        previous_block_activation = x

    x = keras.layers.SeparableConv2D(
        36,
        kernel_size,
        padding="same",
        depthwise_initializer=conv_initializer,
        pointwise_initializer=conv_initializer,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Reshape(target_shape=(-1, x.shape[-2] * x.shape[-1]))(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units,
            kernel_initializer=lstm_initializer,
            recurrent_initializer="orthogonal",
            kernel_regularizer=keras.regularizers.l2(0.001),
            return_sequences=True,
        )
    )(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units,
            kernel_initializer=lstm_initializer,
            recurrent_initializer="orthogonal",
            kernel_regularizer=keras.regularizers.l2(0.001),
            return_sequences=True,
        )
    )(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(
        128,
        activation="relu",
        kernel_initializer=conv_initializer,
        kernel_regularizer=keras.regularizers.l2(0.001),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_labels, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


@keras.saving.register_keras_serializable(name="MaskedBinaryCrossentropy")
class MaskedBinaryCrossentropy(keras.losses.Loss):
    """Masked Binary Crossentropy loss function for Keras"""

    def __init__(
        self,
        mask_value=MASK_VALUE,
        from_logits=False,
        name="MBCE",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.mask_value = mask_value
        self.from_logits = from_logits
        self.loss_fn = keras.losses.BinaryCrossentropy(
            from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, tf.cast(self.mask_value, y_true.dtype))

        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)

        loss = self.loss_fn(y_true_masked, y_pred_masked)

        return tf.reduce_mean(loss)


@keras.saving.register_keras_serializable(name="MaskedBinaryAccuracy")
class MaskedBinaryAccuracy(keras.metrics.BinaryAccuracy):
    """Masked Binary Accuracy metric for Keras"""

    def __init__(self, mask_value=MASK_VALUE, name="MBA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mask_value = mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, tf.cast(self.mask_value, y_true.dtype))
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        return super().update_state(y_true, y_pred, sample_weight)


@keras.saving.register_keras_serializable(name="MaskedAUC")
class MaskedAUC(keras.metrics.AUC):
    """Masked Area Under the Curve (AUC) of ROC metric for Keras"""

    def __init__(self, mask_value=MASK_VALUE, name="MAUC", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mask_value = mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Mask out invalid values
        mask = tf.not_equal(y_true, tf.cast(self.mask_value, y_true.dtype))

        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        return super().update_state(y_true, y_pred, sample_weight)


ORCAI_ARCHITECTURES_FN = {
    "ResNet1DConv": res_net_1Dconv_arch,
    "ResNetLSTM": res_net_LSTM_arch,
}

ORCAI_ARCHITECTURES = list(ORCAI_ARCHITECTURES_FN.keys())


# build model from a choice of models
def build_model(
    input_shape: tuple[int, int, int],
    orcai_parameter: dict,
    msgr: Messenger = Messenger(),
) -> keras.Model:
    """

    Parameters
    ----------
    input_shape : tuple (int, int, int)
        Dimensions of the input data
    orcai_parameter : dict
        OrcAI parameter dictionary
    msgr : Messenger
        Messenger object for messages

    Returns
    -------
        model : keras.Model
            Model
    """
    num_labels = len(orcai_parameter["calls"])
    if orcai_parameter["architecture"] in ORCAI_ARCHITECTURES:
        model = ORCAI_ARCHITECTURES_FN[orcai_parameter["architecture"]](
            input_shape, num_labels, **orcai_parameter["model"]
        )
    else:
        raise ValueError(
            f"Unknown model architecture: {orcai_parameter['architecture']}"
        )

    n_filters = len(orcai_parameter["model"]["filters"])
    output_shape = (input_shape[0] // 2**n_filters, num_labels)

    msgr.part("Building model architecture")
    msgr.info(f"model name:          {orcai_parameter['name']}")
    msgr.info(f"model architecture:  {orcai_parameter['architecture']}")
    msgr.info(f"model input shape:   {model.input_shape}")
    msgr.info(f"model output shape:  {model.output_shape}")
    msgr.info(f"actual input_shape:  {input_shape}")
    msgr.info(f"actual output_shape: {output_shape}")
    msgr.info(f"n_filters:           {n_filters}")
    msgr.info(f"num_labels:          {num_labels}")
    return model
