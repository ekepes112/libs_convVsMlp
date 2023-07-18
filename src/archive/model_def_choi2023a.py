from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, Dropout, MaxPool1D
from keras.regularizers import l1_l2
from warnings import warn
from keras.initializers import HeNormal


def compile_model(
    l1_dense: float = 0.,
    l2_dense: float = 0.,
    kernel_count: int = 16,  # counted in the image, guesswork
    kernel_size: int = 3,
    input_shape: tuple = (None,),
    lr: float = 3e-4,
    optimizer: opt.Optimizer = None,
    loss_func: str = None,
    eval_metrics: list = [],
    model_id: str = 'prototype'
):
    model_id = f'choi2023a_{model_id}'
    if optimizer is None:
        optimizer = opt.Adam(learning_rate=lr)

    model_input = Input(
        shape=input_shape
    )

    x = Conv1D(
        filters=kernel_count,
        kernel_size=kernel_size,
        strides=1,  # not shared
        activation='relu'  # not shared
    )(model_input)
    x = Conv1D(
        filters=kernel_count*2,  # not shared
        kernel_size=kernel_size,
        strides=1,  # not shared
        activation='relu'  # not shared
    )(x)
    x = MaxPool1D(
        pool_size=2  # not shared
    )(x)
    x = Conv1D(
        filters=kernel_count*4,  # not shared
        kernel_size=kernel_size,
        strides=1,  # not shared
        activation='relu'
    )(x)
    x = Flatten()(x)
    x = Dense(
        units=100,  # not shared
        activation='relu'  # not shared
    )(x)
    x = Dropout(.5)(x)  # not shared, .5 taken from another paper

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
