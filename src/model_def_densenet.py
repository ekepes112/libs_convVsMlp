from tensorflow import optimizers as opt
from keras.models import Model
from keras.layers import Input, MaxPool1D, Conv1D, AveragePooling1D,\
    BatchNormalization, GlobalAveragePooling1D, Activation

from residual_modules import residual_block_concat
from ann_modules import prediction_head


class DenseNet1D():
    # https://arxiv.org/pdf/1608.06993v5.pdf
    def __init__(
        self,
        l1_dense: float = 0.,
        l2_dense: float = 0.,
        input_shape: tuple = (None,),
        optimizer: opt.Optimizer = None,
        loss_func: str = None,
        eval_metrics: list = [],
        model_id: str = 'prototype',
        prediction_sizes: list = [2048, 1024],
    ):
        if loss_func is None:
            raise ValueError('No loss function specified')
        if optimizer is None:
            print('No optimizer provided, adding default Adam with 3e-4 initial lr')
            self.optimizer = opt.Adam(learning_rate=3e-4)
        else:
            self.optimizer = optimizer
        self.l1_dense = l1_dense
        self.l2_dense = l2_dense
        self.input_shape = input_shape
        self.loss_func = loss_func
        self.eval_metrics = eval_metrics
        self.model_id = model_id
        self.prediction_sizes = prediction_sizes
        self.model_id = f'densenet_{model_id}'

    def build(self):
        model_input = Input(shape=self.input_shape)
        x = Conv1D(
            kernel_size=7,
            strides=2,
            filters=32,
        )(model_input)
        x = MaxPool1D(
            pool_size=3,
            strides=2,
        )(x)
        for _ in range(4):
            for _ in range(3):
                x = residual_block_concat(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv1D(
                kernel_size=1,
                strides=1,
                filters=32,
            )(x)
            x = AveragePooling1D(
                pool_size=2,
                strides=2,
            )(x)
        for _ in range(3):
            x = residual_block_concat(x)
        # x = GlobalAveragePooling1D()(x)

        output = prediction_head(
            x,
            layer_sizes=self.prediction_sizes,
            lambda_l1=self.l1_dense,
            lambda_l2=self.l2_dense,
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
