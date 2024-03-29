import argparse
import ast
import numpy as np
import pandas as pd
import gc

from pathlib import Path
import shutil

from tensorflow import Variable
from tensorflow.keras.models import clone_model
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session
import wandb
from wandb.keras import WandbCallback

import model_loader
import config
import cv_utils
import config_cv_optimizers
import optimizer_dispatcher


def train_run(
    model_name: str,
    fold: int,
    compound: str,
    predictors_path: str,
    targets_path: str,
    start_checkpoint_at: float,
    checkpoint_dir: str,
    callbacks: list,
    **kwargs
):
    """TODO
        make a config loader -> unique experiment id
    """
    # prepare callbacks
    selected_callbacks = [x.strip() for x in callbacks.split(',')]
    callbacks = [cv_utils.ReinitializeWeights()]
    # load the data
    targets = pd.read_pickle(targets_path)
    predictors = pd.read_pickle(predictors_path)
    # prepare model parameters
    model_params = config.MODEL_PARAMS.get(model_name).copy()
    model_params.update(
        config.SHARED_MODEL_PARAMS
    )
    model_params.update({
        'input_shape': (predictors.shape[1], 1),
        'model_id': 'training',
    })
    model_params.update(kwargs)
    # define model architecture
    print(f'processing:: {model_name} - {fold}')
    base_model = model_loader.models.get(
        model_name,
        'Invalid model name'
    )(
        **model_params,
    ).build()
    # split the data names
    train_names = targets.loc[
        targets.loc[:, f'{compound}_Folds'] != fold, :
    ].index
    val_names = targets.loc[
        targets.loc[:, f'{compound}_Folds'] == fold, :
    ].index
    # get optimal optimizer
    optimizer = optimizer_dispatcher.generate_optimizer(
        **config_cv_optimizers.OPTIMIZER_PARAMS[
            config.OPTIMIZERS.get(base_model.name)
        ]
    )
    optimizer.learning_rate = Variable(
        config.INITIAL_LEARNING_RATES.get(base_model.name)
    )
    # initialize w&b logging
    if 'wandb' in selected_callbacks:
        wandb_run = wandb.init(
            project=config.PROJECT_NAME,
            name=f'{base_model.name}',
            notes=f'fold_{fold} with weight reinit.'
        )
        wandb_run.define_metric(
            name='val_root_mean_squared_error',
            summary='min',
            goal='minimize'
        )
        wandb_run.define_metric(
            name='val_root_mean_squared_error',
            summary='last',
            goal='minimize'
        )

        callbacks.append(
            WandbCallback(
                log_weights=False,
                save_model=False
            )
        )
    # clone architecture to reset weights
    model = clone_model(base_model)
    model.compile(
        optimizer=optimizer,
        loss=model_params.get('loss_func'),
        metrics=[
            RootMeanSquaredError(),
            MeanAbsoluteError()
        ]
    )
    # create checkpoint directory
    if 'checkpointing' in selected_callbacks:
        checkpoint_path = checkpoint_dir.joinpath(
            f'{model.name}_fold_{fold}/cp_lowest_validation_rmse.ckpt'
        )
        if not checkpoint_path.parent.is_dir():
            print('creating directory')
            checkpoint_path.parent.mkdir(parents=True)
        else:
            print('checkpoint directory already exists')
        # initialize checkpoint callback
        callback_checkpointing = ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1,
            monitor='val_root_mean_squared_error',
            mode='min',
            save_best_only=True,
            initial_value_threshold=start_checkpoint_at
        )
        callbacks.append(callback_checkpointing)
    # fit the model
    model.fit(
        x=predictors.loc[train_names, :]
        .to_numpy()[..., np.newaxis],
        y=targets.loc[train_names, compound],
        batch_size=config.BATCH_SIZE,
        epochs=config.TRAIN_EPOCHS,
        validation_data=(
            predictors.loc[val_names, :]
            .to_numpy()[..., np.newaxis],
            targets.loc[val_names, compound]
        ),
        callbacks=callbacks
    )

    model_path = checkpoint_dir.parent.joinpath('models')\
        .joinpath(f'{model.name}_fold_{fold}_fin')
    model.save(filepath=model_path)
    shutil.move(
        checkpoint_path.parent,
        model_path.with_name(
            model_path.name.replace(
                'fin',
                'checkpoint'
            )
        )
    )

    # clean up
    del model
    del optimizer
    clear_session()
    gc.collect()


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
        '--fold',
        type=int,
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
        '--callbacks',
        type=str,
    )
    argument_parser.add_argument(
        '--kwargs',
        type=str,
    )

    cmd_args = argument_parser.parse_args()

    train_run(
        model_name=cmd_args.model,
        fold=cmd_args.fold,
        compound=cmd_args.compound,
        predictors_path=cmd_args.predictors,
        targets_path=cmd_args.targets,
        start_checkpoint_at=6.,
        checkpoint_dir=Path(config.CHECKPOINT_DIR),
        callbacks=cmd_args.callbacks,
        **ast.literal_eval(cmd_args.kwargs)
    )