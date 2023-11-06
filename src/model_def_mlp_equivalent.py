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
        """
        Initializes the MLP model with the given parameters.

        Args:
            input_shape (tuple): The shape of the input data. Default is (None,).
            hidden_layer_size (int): The size of the hidden layer. Default is 2048.
            optimizer (opt.Optimizer): The optimizer to be used for training. Default is None.
            loss_func (str): The loss function to be used for training. Default is None.
            eval_metrics (list): The evaluation metrics to be used for evaluation. Default is an empty list.
            model_id (str): The ID of the model. Default is 'prototype'.
            dropout_rate (float): The dropout rate to be used. Default is 0.
            prediction_head_params (dict): The parameters for the prediction head. Default is an empty dictionary.

        Raises:
            ValueError: If no loss function is specified.

        Returns:
            None
        """
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
        """
        Builds and compiles the model.

        Returns:
            model (Model): The compiled neural network model.
        """
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