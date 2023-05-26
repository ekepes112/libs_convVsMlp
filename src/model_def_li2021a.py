from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, BatchNormalization, Activation, GlobalAveragePooling1D, MaxPool1D
from keras.regularizers import l1_l2
from warnings import warn
from keras.initializers import HeNormal
from modul_def_residual import residual_block


def compile_model(
    l1_dense: float = 0.,  # not mentioned
    l2_dense: float = 0.,  # not mentioned
    input_shape: tuple = (None,),
    lr: float = 3e-4,
    optimizer: opt.Optimizer = None,
    loss_func: str = None,
    eval_metrics: list = [],
    model_id: str = 'prototype',
    add_batch_norm: bool = True
):
    model_id = f'li2021a_{model_id}'
    if optimizer is None:
        optimizer = opt.Adam(learning_rate=lr)

    model_input = Input(shape=input_shape)

    x = Conv1D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same'
    )(model_input)
    if add_batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = residual_block(x, kernel_counts=64, add_batch_norm)
    x = residual_block(x, kernel_counts=64, add_batch_norm)
    x = residual_block(x, kernel_counts=128, add_batch_norm)
    x = residual_block(x, kernel_counts=128, add_batch_norm)
    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)

    x = Dense(
        units=1048,
        activation='relu',
    )(x)

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
