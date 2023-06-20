from keras.layers import Conv1D, BatchNormalization, Activation, Add


def residual_block_simple(
    input_layer,
    filters: int,
    kernel_size: int = 3,
):
    x = Activation('relu')(input_layer)
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
    )(x)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=input_layer.shape[-1],
        kernel_size=kernel_size,
        strides=1,
        padding='same'
    )(x)
    x = Add()([input_layer, x])
    return x


def residual_block_bn(
    input_layer,
    filters: int,
    kernel_size: int = 3,
):
    x = BatchNormalization()(input_layer)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=input_layer.shape[-1],
        kernel_size=kernel_size,
        strides=1,
        padding='same'
    )(x)
    x = Add()([input_layer, x])
    return x


def residual_block_bottleneck(
    input_layer,
    filters: int,
    kernel_size: int = 3,
):
    x = BatchNormalization()(input_layer)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=input_layer.shape[-1],
        kernel_size=1,
        strides=1,
        padding='same'
    )(x)
    x = Add()([input_layer, x])
    return x


def residual_block_concat(
    input_layer,
    filters: int,
    kernel_size: int = 3,
):
    x = BatchNormalization()(input_layer)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=input_layer.shape[-1],
        kernel_size=1,
        strides=1,
        padding='same'
    )(x)
    x = Add()([input_layer, x])
    return x
