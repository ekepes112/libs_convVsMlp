import argparse
import ast
import numpy as np
import pandas as pd
from pathlib import Path

from tensorflow.keras.models import clone_model
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

import model_loader
import config


def evaluate_model(
    model_name:str,
    weights_path:str,
    compound: str,
    predictors_path: str,
    targets_path: str,
    **kwargs
):
    """
    Evaluates a trained model using the specified parameters.
    The results of the evaluation are printed in the console.

    Args:
        model_name (str): The name of the model to be evaluated.
        weights_path (str): The path to the weights file of the trained model.
        compound (str): The compound to evaluate the model on.
        predictors_path (str): The path to the file containing the predictors.
        targets_path (str): The path to the file containing the targets.
        **kwargs: Additional keyword arguments to be passed to the model.

    Returns:
        None
    """
    print('##############################################################')
    print(f'processing:: {model_name}')
    # load the data
    targets = pd.read_pickle(targets_path)
    predictors = pd.read_pickle(predictors_path)
    # prepare model parameters
    model_params = config.MODEL_PARAMS.get(model_name).copy()
    model_params.update(
        config.SHARED_MODEL_PARAMS
    )
    model_params.update({
        'input_shape':(predictors.shape[1],1)
    })
    model_params.update(kwargs)
    # define model architecture
    model = model_loader.models.get(
        model_name,
        'Invalid model name'
    )(
        **model_params
    ).build()
    # clone architecture to reset weights
    model.compile(
        loss=model_params.get('loss_func'),
        metrics=[
            RootMeanSquaredError(),
            MeanAbsoluteError()
        ]
    )
    # get weights of selected trained model
    model.load_weights(weights_path)

    print(f'performance on {predictors_path}')
    model.evaluate(
        x=predictors.to_numpy()[...,np.newaxis],
        y=targets.loc[:,compound]
    )
    return None


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--model',
        type=str,
    )
    argument_parser.add_argument(
        '--compound',
        type=str,
    )
    argument_parser.add_argument(
        '--targets',
        type=str,
    )
    argument_parser.add_argument(
        '--predictors',
        type=str,
    )
    argument_parser.add_argument(
        '--weights',
        type=str,
    )
    argument_parser.add_argument(
        '--kwargs',
        type=str,
        default='{}',
    )
    cmd_args = argument_parser.parse_args()

    evaluate_model(
        model_name=cmd_args.model,
        compound=cmd_args.compound,
        weights_path=cmd_args.weights,
        predictors_path=cmd_args.predictors,
        targets_path=cmd_args.targets,
        **ast.literal_eval(cmd_args.kwargs)
    )