import argparse
import ast
import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow import Variable
from tensorflow.keras.models import clone_model
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import ModelCheckpoint
import wandb
from wandb.keras import WandbCallback

import model_loader
import config
import cv_utils
import config_cv_optimizers
import learning_rate_estimator
import optimizer_dispatcher


def cv_run(
    model_name: str,
    fold: int,
    compound: str,
    predictors_path: str,
    targets_path: str,
    lr_scan_params: dict,
    callbacks: list,
    checkpoint_dir: str,
    start_checkpoint_at: float,
    **kwargs
):
    # prepare callbacks
    selected_callbacks = [x.strip() for x in callbacks.split(',')]
    callbacks = [cv_utils.ReinitializeWeights()]
    # load the data
    predictors = pd.read_pickle(predictors_path)
    targets = pd.read_pickle(targets_path)
    # prepare model parameters
    model_params = config.MODEL_PARAMS.get(model_name).copy()
    model_params.update(
        config.SHARED_MODEL_PARAMS
    )
    model_params.update({
        'input_shape': (predictors.shape[1], 1),
        'model_id': 'optimizer_cv_',
    })
    model_params.update(kwargs)
    # define model architecture
    # base_model = model_loader.models.get(
    #     model_name,
    #     'Invalid model name'
    # )(
    #     **model_params,
    # ).build()
    print(f'processing fold {fold}')
    # split the data names
    train_names = targets.loc[
        targets.loc[:, f'{compound}_Folds'] != fold, :
    ].index
    val_names = targets.loc[
        targets.loc[:, f'{compound}_Folds'] == fold, :
    ].index
    # update the parameters for the learning rate estimation with the split data
    lr_scan_params['train_data'] = (
        predictors.loc[train_names, :].to_numpy()[..., np.newaxis],
        targets.loc[train_names, compound],
    )
    # estimate the initial learning rate
    for optimizer_name in config_cv_optimizers.OPTIMIZERS:
        learning_rate_estimator.estimate_learnig_rate(
            base_model=model_loader.models.get(
                cmd_args.model,
                'Invalid model name'
            )(
                **model_params,
            ).build(),
            optimizer=optimizer_dispatcher.generate_optimizer(
                optimizer_name,
                config_cv_optimizers.OPTIMIZER_PARAMS.get(optimizer_name),
            ),
            **lr_scan_params
        )
    # loop over each explored optimizer
    lr_estimates_dir = Path(lr_scan_params['results_path'])\
        .joinpath('lr_estimates')\
        .joinpath(f"model_name_{model_params['model_id']}")
        # .joinpath(base_model.name)
    for file_path in lr_estimates_dir.glob('*.txt'):
        with open(file_path, 'r') as file:
            lr_estimate = float(file.read())
        optimizer_name = '_'.join(
            file_path.name.split('_')[-2:]
        ).replace('.txt', '')
        print(f'{optimizer_name}:: {lr_estimate:.4f}')
        # take the current optimizer
        optimizer = optimizer_dispatcher.generate_optimizer(
            optimizer_name,
            config_cv_optimizers.OPTIMIZER_PARAMS.get(optimizer_name),
        )
        if not optimizer:
            continue
        optimizer.learning_rate = Variable(lr_estimate)
        # initialize w&b logging
        if 'wandb' in selected_callbacks:
            wandb_run = wandb.init(
                project=config.PROJECT_NAME,
                name=f"{model_name}{model_params['model_id']}{optimizer_name}",
                notes=f'fold_{fold}; {lr_estimate:.2e}, with weight reinit.'
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
        model = model_loader.models.get(
            model_name,
            'Invalid model name'
        )(
            **model_params,
        ).build()
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
                f'{model.name}{optimizer_name}_fold_{fold}/cp_lowest_validation_rmse.ckpt'
            )
            if not checkpoint_path.parent.is_dir():
                print('creating direction')
                checkpoint_path.parent.mkdir()
            else:
                print('creating direction')
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
            epochs=config_cv_optimizers.CV_EPOCHS,
            validation_data=(
                predictors.loc[val_names, :]
                .to_numpy()[..., np.newaxis],
                targets.loc[val_names, compound]
            ),
            callbacks=callbacks
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
        '--folds',
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
        '--callbacks',
        type=str,
    )
    argument_parser.add_argument(
        '--kwargs',
        type=str,
    )
    cmd_args = argument_parser.parse_args()

    for fold in cmd_args.folds.split(','):
        cv_run(
            model_name=cmd_args.model,
            fold=fold,
            compound=cmd_args.compound,
            predictors_path=cmd_args.predictors,
            targets_path=cmd_args.targets,
            lr_scan_params={
                'batch_size': config.BATCH_SIZE,
                'results_path': Path(config_cv_optimizers.RESULTS_PATH),
                'loss_function':config.SHARED_MODEL_PARAMS['loss_func'],
                'end_lr': config_cv_optimizers.LR_SCAN_END,
                'step_size': config_cv_optimizers.LR_SCAN_STEP_SIZE,
                'warmup_count': config_cv_optimizers.LR_SCAN_WARMUP,
                'clone_model': True,
                'overwrite_existing': True,
                'return_data': False,
                'training_verbosity': 0,
                'save_fig': False,
            },
            callbacks=cmd_args.callbacks,
            checkpoint_dir=Path(config.CHECKPOINT_DIR),
            start_checkpoint_at=5.,
            **ast.literal_eval(cmd_args.kwargs),
        )