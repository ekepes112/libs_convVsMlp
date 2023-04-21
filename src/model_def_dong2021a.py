from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPool1D
from keras.regularizers import l1_l2
from warnings import warn
from keras.initializers import HeNormal
from modul_def_inception import inception_module


def compile_model(
    l1_dense: float = 0.,
    l2_dense: float = 0.,
    kernel_counts: list = [8],
    kernel_sizes: list = [],
    conv_strides: list = [],
    pooling_sizes: list = [],
    input_shape: tuple = (None,),
    lr: float = 3e-4,
    optimizer: opt.Optimizer = None,
    loss_func: str = None,
    eval_metrics: list = [],
    model_id: str = 'prototype'
):
    model_id = f'dong2021a_{model_id}'

    if optimizer is None:
        optimizer = opt.Adam(learning_rate=lr)

    model_input = Input(
        shape=input_shape
    )

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
        x = MaxPool1D(
            pool_size=ps
        )(x)
    x = inception_module(x)
    x = Flatten()(x)
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
