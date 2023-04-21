from keras.layers import Conv1D, Concatenate, MaxPool1D
from keras.initializers import HeNormal
from keras.regularizers import l2


def inception_module(
    input_layer,
    kernel_count: int = 4,
    act_func: str = 'tanh',
    l2_norm: float = 0.
):
    ###############
    branch_1 = Conv1D(
        filters=kernel_count,
        kernel_size=1,
        activation='linear',
        kernel_initializer=HeNormal,
        kernel_regularizer=l2(
            l2=l2_norm
        ),
        padding='same'
    )(input_layer)
    ###############
    branch_2 = Conv1D(
        filters=kernel_count,
        kernel_size=1,
        activation='linear',
        kernel_initializer=HeNormal,
        kernel_regularizer=l2(
            l2=l2_norm
        ),
        padding='same'
    )(input_layer)
    branch_2 = Conv1D(
        filters=kernel_count,
        kernel_size=3,
        activation=act_func,
        kernel_initializer=HeNormal,
        kernel_regularizer=l2(
            l2=l2_norm
        ),
        padding='same'
    )(branch_2)
    ###############
    branch_3 = Conv1D(
        filters=kernel_count,
        kernel_size=1,
        activation='linear',
        kernel_initializer=HeNormal,
        kernel_regularizer=l2(
            l2=l2_norm
        ),
        padding='same'
    )(input_layer)
    branch_3 = Conv1D(
        filters=kernel_count,
        kernel_size=5,
        activation=act_func,
        kernel_initializer=HeNormal,
        kernel_regularizer=l2(
            l2=l2_norm
        ),
        padding='same'
    )(branch_3)
    ###############
    branch_4 = MaxPool1D(
        pool_size=2,
        padding='same',
        strides=1
    )(input_layer)
    branch_4 = Conv1D(
        filters=kernel_count,
        kernel_size=1,
        activation=act_func,
        kernel_initializer=HeNormal,
        kernel_regularizer=l2(
            l2=l2_norm
        ),
        padding='same'
    )(branch_4)

    out = Concatenate()([branch_1, branch_2, branch_3, branch_4])

    return (out)
