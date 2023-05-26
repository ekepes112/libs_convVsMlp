from tensorflow import optimizers as opt
from tensorflow.random import uniform
from tensorflow import dtypes, concat, zeros, ones
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, BatchNormalization, Activation, GlobalAveragePooling1D, MaxPool1D, Layer
from keras.regularizers import l1_l2
from warnings import warn
from keras.initializers import HeNormal
from modul_def_residual import residual_block


class RandomMask(Layer):

    def __init__(self, size, **kwargs):
        super(RandomMask, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, training):
        if not training:
            return inputs
        index = uniform(
            shape=[],
            minval=0,
            maxval=inputs.shape[1]-self.size,
            dtype=dtypes.int32,
            seed=None,
            name=None
        )
        mask = concat(
            [ones(
                [inputs.shape[0], index],
                dtype=dtypes.float32
            ),
                zeros(
                [inputs.shape[0], self.size],
                dtype=dtypes.float32
            ),
                ones(
                [inputs.shape[0], inputs.shape[1] - index - self.size],
                dtype=dtypes.float32
            )],
            axis=1
        )

        return inputs * mask


def compile_model(
    l1_dense: float = 0.,  # not mentioned
    l2_dense: float = 0.,  # not mentioned
    input_shape: tuple = (None,),
    lr: float = 3e-4,
    optimizer: opt.Optimizer = None,
    loss_func: str = None,
    eval_metrics: list = [],
    model_id: str = 'prototype',
    add_batch_norm: bool = True,
    mask_size: int = 1000
):
    model_id = f'li2021a_{model_id}'
    if optimizer is None:
        optimizer = opt.Adam(learning_rate=lr)

    model_input = Input(shape=input_shape)
    model_input = RandomMask(size=mask_size)(model_input)

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
    x = residual_block(x, kernel_counts=64, add_batch_norm=add_batch_norm)
    x = residual_block(x, kernel_counts=64, add_batch_norm=add_batch_norm)
    x = residual_block(x, kernel_counts=128, add_batch_norm=add_batch_norm)
    x = residual_block(x, kernel_counts=128, add_batch_norm=add_batch_norm)
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
