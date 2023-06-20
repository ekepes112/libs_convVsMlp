from keras.layers import Dense, Flatten
from keras.regularizers import l1_l2
from keras.initializers import HeNormal


def prediction_head(
    input_layer,
    layer_sizes: list = [2048, 1024],
    lambda_l1: float = 0.,
    lambda_l2: float = 0.,
):
    x = Flatten()(input_layer)
    for l in layer_sizes:
        x = Dense(
            l,
            activation='relu',
            kernel_initializer=HeNormal,
            kernel_regularizer=l1_l2(
                l1=lambda_l1,
                l2=lambda_l2
            )
        )(x)
    x = Dense(
        1,
        activation='relu',
        kernel_initializer=HeNormal,
        kernel_regularizer=l1_l2(
            l1=lambda_l1,
            l2=lambda_l2
        )
    )(x)
    return x
