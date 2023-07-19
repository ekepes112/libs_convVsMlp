from keras.layers import Dense, Flatten, Dropout
from keras.regularizers import l1_l2
from keras.initializers import HeNormal


def prediction_head(
    input_layer,
    layer_sizes: list = [2048, 1024],
    lambda_l1: float = 0.,
    lambda_l2: float = 0.,
    dropout_rate: float = 0.,
):
    """
    Generates a prediction module for a neural network model.

    Args:
        input_layer (_type_): The input layer of the prediction module (the last of the preceding layers).
        layer_sizes (list, optional): The sizes of the hidden layers in the prediction module. Defaults to [2048, 1024].
        lambda_l1 (float, optional): The L1 regularization parameter. Defaults to 0..
        lambda_l2 (float, optional): The L2 regularization parameter. Defaults to 0..
        dropout_rate (float, optional): The dropout rate to be applied to each of the module's dropout layers. Defaults to 0..

    Returns:
        _type_: The prediction module for the neural network model.
    """
    prediction_module = Flatten()(input_layer)
    # Iterate over the layer sizes and add dropout and dense layers
    for layer_size in layer_sizes:
        prediction_module = Dropout(dropout_rate)(prediction_module)
        prediction_module = Dense(
            layer_size,
            activation='relu',
            kernel_initializer=HeNormal,
            kernel_regularizer=l1_l2(l1=lambda_l1, l2=lambda_l2)
        )(prediction_module)

    # Add the final dropout and dense layer
    prediction_module = Dropout(dropout_rate)(prediction_module)
    prediction_module = Dense(
        1,
        activation='relu',
        kernel_initializer=HeNormal,
        kernel_regularizer=l1_l2(l1=lambda_l1, l2=lambda_l2)
    )(prediction_module)

    return prediction_module
