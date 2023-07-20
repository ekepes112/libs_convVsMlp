from tensorflow.keras.optimizers import Adam, SGD

def generate_optimizer(
    optimizer_type,
    **kwargs
):
    if optimizer_type.lower() == 'adam':
        return Adam(
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        return SGD(
            **kwargs
        )