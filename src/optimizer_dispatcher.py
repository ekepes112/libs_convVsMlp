from tensorflow.keras.optimizers import Adam, SGD

def generate_optimizer(
    optimizer_type,
    **kwargs
):
    """
    Generates an optimizer based on the given optimizer type and keyword arguments.
    A convenience function for running cross-validation with different optimizers.

    Parameters:
        optimizer_type (str): The type of optimizer to generate. Can be 'adam' or 'sgd'.
        **kwargs: Additional keyword arguments to pass to the optimizer constructor.

    Returns:
        The generated optimizer object.

    Raises:
        ValueError: If an invalid optimizer type is provided.
    """
    if optimizer_type.lower() == 'adam':
        return Adam(
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        return SGD(
            **kwargs
        )