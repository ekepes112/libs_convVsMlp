from keras.layers import Conv1D, BatchNormalization, Activation, Add


def residual_block(
    input_layer,
    kernel_counts: int,
    add_batch_norm: bool = True
):
    x = Conv1D(
        filters=kernel_counts,
        kernel_size=3,
        strides=1,
        padding='same'
    )(input_layer)
    if add_batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(
        filters=kernel_counts,
        kernel_size=3,
        strides=1,
        padding='same'
    )(x)
    if add_batch_norm:
        x = BatchNormalization()(x)

    if input_layer.shape[2] != x.shape[2]:
        input_layer = Conv1D(
            filters=kernel_counts,
            kernel_size=1,
            strides=1
        )(input_layer)

    x = Add()([input_layer, x])
    x = Activation('relu')(x)

    return (x)
