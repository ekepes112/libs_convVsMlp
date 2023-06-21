from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPool1D
from warnings import warn

from ann_modules import prediction_head


def compile_model(
    l1_dense: float = 0.,
    l2_dense: float = 0.,
    input_shape: tuple = (None,),
    optimizer: opt.Optimizer = None,
    loss_func: str = None,
    eval_metrics: list = [],
    model_id: str = 'prototype',
    prediction_sizes: list = [2048, 1024],
):
    if optimizer is None:
        optimizer = opt.Adam(learning_rate=3e-4)
    if loss_func is None:
        raise ValueError('No loss function specified')

    conv_layer_counts = [2, 4, 4, 4]
    conv_layer_sizes = [64, 128, 256, 256]

    model_input = Input(shape=input_shape)
    x = Conv1D(
        filters=32,  # lowered by a factor of 3 compared to the original work since we have 3 times fewer channels
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
    )(model_input)
    x = Conv1D(
        filters=32,  # lowered by a factor of 3 compared to the original work since we have 3 times fewer channels
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
    )(x)
    x = MaxPool1D(
        pool_size=2,
    )(x)
    for conv_layer_count, conv_layer_size in zip(conv_layer_counts, conv_layer_sizes):
        for _ in range(conv_layer_count):
            x = Conv1D(
                filters=conv_layer_size,  # see motivation above, the scaling is not exact
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
            )(x)
        x = MaxPool1D(
            pool_size=2,
        )(x)

    output = prediction_head(
        x,
        layer_sizes=prediction_sizes,
        lambda_l1=l1_dense,
        lambda_l2=l2_dense,
    )

    model = Model(
        model_input,
        output,
        name=model_id
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=eval_metrics
    )

    return model
