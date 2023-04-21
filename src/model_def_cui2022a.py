from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, Dropout, BatchNormalization, MaxPool1D
from keras.regularizers import l1_l2
from warnings import warn
from keras.initializers import HeNormal


def compile_model(
    l1_dense: float = 0.,
    l2_dense: float = 0.,
    kernel_count: int = 32,
    kernel_size: int = 25,
    input_shape: tuple = (None,),
    lr: float = 3e-4,
    optimizer: opt.Optimizer = None,
    loss_func: str = None,
    eval_metrics: list = [],
    model_id: str = 'prototype'
):
    model_id = f'cui2022a_{model_id}'
    if optimizer is None:
        optimizer = opt.Adam(learning_rate=lr)

    model_input = Input(shape=input_shape)

    x = BatchNormalization()(model_input)
    x = Conv1D(
        filters=kernel_count,
        kernel_size=kernel_size,
        strides=1,
        activation='relu'
    )(x)
    x = MaxPool1D(
        pool_size=2
    )(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(
        units=512,
        activation='relu'
    )(x)
    x = Dropout(.2)(x)  # .2 in paper, layer's location not described
    x = Dense(
        units=128,
        activation='relu'
    )(x)
    x = Dropout(.2)(x)  # .2 in paper, layer's location not described
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
