from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, Conv1D, Maxpool1D

from residual_modules import residual_block_concat
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
    model_id = f'densenet_{model_id}'
    if optimizer is None:
        optimizer = opt.Adam(learning_rate=3e-4)
    if loss_func is None:
        raise ValueError('No loss function specified')

    model_input = Input(shape=input_shape)

    x = Maxpool1D(pool_size=2)(model_input)
    for _ in range(3):
        for _ in range(3):
            x = residual_block_concat(x)
        x = Maxpool1D(pool_size=2)(x)

    # x = Conv1D(
    #     filters=1,
    #     kernel_size=1,
    #     strides=1,
    #     padding='same',
    #     activation='relu'
    # )(x)

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
