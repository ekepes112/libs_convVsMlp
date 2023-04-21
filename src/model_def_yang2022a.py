from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, Dropout, MaxPool1D
from keras.regularizers import l1_l2
from warnings import warn
from keras.initializers import HeNormal


def compile_model(
    l1_dense: float = 0.,  # not mentioned
    l2_dense: float = 0.,  # not mentioned
    kernel_counts: list = [8, 32, 128, 256, 512],
    kernel_sizes: list = [5, 5, 5, 5, 5],
    conv_strides: list = [2, 2, 2, 2, 2],
    pooling_sizes: list = [2, 2, 0, 0, 0],
    input_shape: tuple = (None,),
    lr: float = 3e-4,
    optimizer: opt.Optimizer = None,
    loss_func: str = None,
    eval_metrics: list = [],
    model_id: str = 'prototype'
):
    model_id = f'yang2022a_{model_id}'

    if optimizer is None:
        optimizer = opt.Adam(learning_rate=lr)

    model_input = Input(shape=input_shape)

    x = model_input
    for kc, ks, cs, ps in zip(
        kernel_counts, kernel_sizes, conv_strides, pooling_sizes
    ):
        x = Conv1D(
            filters=kc,
            kernel_size=ks,
            strides=cs,
            activation='relu'
        )(x)
        if ps != 0:
            x = MaxPool1D(
                pool_size=ps
            )(x)
    x = Flatten()(x)
    model_output = Dense(
        1024,
        activation='relu',
        kernel_initializer=HeNormal,
        kernel_regularizer=l1_l2(
            l1=l1_dense,
            l2=l2_dense
        )
    )(x)
    x = Dropout(.2)(x)  # based on other papers
    model_output = Dense(
        1,
        activation='relu',
        kernel_initializer=HeNormal,
        kernel_regularizer=l1_l2(
            l1=l1_dense,
            l2=l2_dense
        )
    )(x)

    model = Model(
        model_input,
        model_output,
        name=model_id
    )

    if not loss_func:
        warn('Model compiled without loss function')

    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=eval_metrics
    )

    return (model)
