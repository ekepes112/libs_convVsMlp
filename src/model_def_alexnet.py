from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPool1D

from ann_modules import prediction_head


class AlexNet1D():
    def __init__(
        self,
        input_shape: tuple = (None,),
        optimizer: opt.Optimizer = None,
        loss_func: str = None,
        eval_metrics: list = [],
        model_id: str = 'prototype',
        prediction_head_params: dict = {},
    ):
        if loss_func is None:
            raise ValueError('No loss function specified')
        if optimizer is None:
            print('No optimizer provided, adding default Adam with 3e-4 initial lr')
            self.optimizer = opt.Adam(learning_rate=3e-4)
        else:
            self.optimizer = optimizer
        self.input_shape = input_shape
        self.loss_func = loss_func
        self.eval_metrics = eval_metrics
        self.model_id = model_id
        self.prediction_head_params = prediction_head_params
        self.model_id = f'alexnet_{model_id}'

    def build(self):
        model_input = Input(shape=self.input_shape)
        x = Conv1D(
            filters=32,  # lowered by a factor of 3 compared to the original work since we have 3 times fewer channels
            kernel_size=11,
            strides=4,
            padding='same',
            activation='relu',
        )(model_input)
        x = MaxPool1D(
            pool_size=2,
        )(x)
        x = Conv1D(
            filters=64,  # see motivation above, the scaling is not exact
            kernel_size=5,
            strides=1,
            padding='same',
            activation='relu',
        )(x)
        x = MaxPool1D(
            pool_size=2,
        )(x)
        x = Conv1D(
            filters=96,  # see motivation above, the scaling is not exact
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
        )(x)
        x = Conv1D(
            filters=96,  # see motivation above, the scaling is not exact
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
        )(x)
        x = Conv1D(
            filters=64,  # see motivation above, the scaling is not exact
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
        )(x)

        output = prediction_head(
            x,
            **self.prediction_head_params,
        )

        model = Model(
            model_input,
            output,
            name=self.model_id
        )

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_func,
            metrics=self.eval_metrics
        )

        return model
