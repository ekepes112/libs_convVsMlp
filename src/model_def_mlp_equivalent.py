from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout

from ann_modules import prediction_head


class MLP():
    def __init__(
        self,
        input_shape: tuple = (None,),
        hidden_layer_size: int = 2048,
        optimizer: opt.Optimizer = None,
        loss_func: str = None,
        eval_metrics: list = [],
        model_id: str = 'prototype',
        dropout_rate: float = 0.,
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
        self.hidden_layer_size = hidden_layer_size
        self.loss_func = loss_func
        self.eval_metrics = eval_metrics
        self.model_id = model_id
        self.dropout_rate = dropout_rate
        self.prediction_head_params = prediction_head_params
        self.model_id = f'mlp_{model_id}'

    def build(self):
        model_input = Input(shape=self.input_shape)
        x = Flatten()(model_input)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(
            units=self.hidden_layer_size
        )(x)
        x = Dropout(self.dropout_rate)(x)

        output = prediction_head(
            x,
            **self.prediction_head_params
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