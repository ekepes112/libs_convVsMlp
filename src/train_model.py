import argparse
import numpy as np
import pandas as pd

from pathlib import Path
import shutil

from tensorflow import Variable
from tensorflow.keras.models import clone_model
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.checkpoints import ModelCheckpoint
import wandb
from wandb.keras import WandbCallback

import model_loader
import config
import cv_utils

def train_run(
    model_name: str,
    fold: int,
    compound: str,
    predictors_path: str,
    targets_path: str,
    start_checkpoint_at: float,
    checkpoint_dir: str,
):
    # load the data
    targets = pd.read_pickle(targets_path)
    predictors = pd.read_pickle(predictors_path)
    # define model architecture
    print(f'processing:: {model_name} - {fold}')
    base_model = model_loader.models.get(
        model_name,
        'Invalid model name'
    )(
        model_id='training',
        **config.MODEL_PARAMS.get(model_name),
        **config.SHARED_MODEL_PARAMS,
        input_shape=(predictors.shape[1],1),
    ).build()
    # take the core model name
    base_model_name = base_model.name.split('_')[0]
    # split the data names
    train_names = targets.loc[
        targets.loc[:,f'{compound}_Folds'] != fold,:
    ].index
    val_names = targets.loc[
        targets.loc[:,f'{compound}_Folds'] == fold,:
    ].index
    # get optimal optimizer
    optimizer = cv_utils.generate_optimizers()[
        config.OPTIMIZERS.get(model_name)
    ]
    optimizer.learning_rate = Variable(
        config.INITIAL_LEARNING_RATES.get(model_name)
    )

    # initialize w&b logging
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
    # clone architecture to reset weights
    model = clone_model(base_model)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[
            RootMeanSquaredError(),
            MeanAbsoluteError()
        ]
    )
    # create checkpoint directory
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
    # fit the model
    model.fit(
        x=predictors.loc[train_names,:]\
          .to_numpy()[...,np.newaxis],
        y=targets.loc[train_names,compound],
        batch_size=config.BATCH_SIZE,
        epochs=config.TRAIN_EPOCHS,
        validation_data=(
            predictors.loc[val_names,:]\
              .to_numpy()[...,np.newaxis],
            targets.loc[val_names,compound]
        ),
        callbacks=[
            callback_checkpointing,
            WandbCallback(
                log_weights=False,
                save_model=False
            ),
            cv_utils.ReinitializeWeights()
        ]
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
    cmd_args = argument_parser.parse_args()

    train_run(
        model_name=cmd_args.model,
        compound=cmd_args.compound,
        fold=cmd_args.fold,
        predictors_path=cmd_args.predictors,
        targets_path=cmd_args.targets,
        start_checkpoint_at=6.,
        checkpoint_dir=Path(config.CHECKPOINT_DIR),
    )