from tensorflow import optimizers as opt
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.regularizers import l1_l2
from warnings import warn

def compile_simple_dense(
    l1_dense:list = [0],
    l2_dense:list = [0],
    dropout_rates:list = [.0, None],
    layer_sizes:list = [2000,1000,500,1],
    activation_functions:list = ['relu'],
    input_shape:tuple = (None,),
    lr:float = 3e-4,
    optimizer:opt.Optimizer = None,
    loss_func:str = None,
    eval_metrics:list = [],
    model_id:str = 'prototype'
):

    l1_dense = extend_parameter_list(layer_sizes,l1_dense)
    l2_dense = extend_parameter_list(layer_sizes,l2_dense)
    dropout_rates = extend_parameter_list(layer_sizes,dropout_rates)
    activation_functions = extend_parameter_list(layer_sizes,activation_functions)

    if optimizer is None:
        optimizer = opt.Adam(learning_rate=lr)

    model = Sequential(name=f'simple_dense_{model_id}')
    model.add(Input(shape=input_shape))

    for units,act_func,l1,l2,do in zip(
        layer_sizes,
        activation_functions,
        l1_dense,
        l2_dense,
        dropout_rates
    ):
        model.add(
            Dense(
                units=units,
                activation=act_func,
                kernel_regularizer=l1_l2(
                    l1=l1,
                    l2=l2
                )
            )
        )
        if do: model.add(Dropout(do))

    if not loss_func: warn('Model compiled without loss function')

    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=eval_metrics
    )

    return(model)


def extend_parameter_list(
    reference_params:list,
    params:list,
) -> list:
    if len(params) == 1:
        return(params * len(reference_params))
    elif len(params) == 2:
        return(([params[0]] * (len(reference_params) - 1)) + [params[1]])
    elif len(reference_params) != len(params) & len(params) >= 3:
        raise ValueError('Lengths of provided parameter lists are not compatible')