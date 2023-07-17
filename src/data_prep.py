import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing
import plotly.graph_objs as go

import data_loaders
import data_preprocessors
import config
import graph_utils


if __name__ == '__main__':
    DRIVE_PATH = Path(config.DRIVE_PATH)
    DATA_PATH = DRIVE_PATH.joinpath(config.DATA_PATH)

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--dataset',
        type=str,
    )
    argument_parser.add_argument(
        '--compound',
        type=str,
    )

    cmd_args = argument_parser.parse_args()

    if cmd_args.dataset == 'supercam':
        import config_supercam
        data, wvl, targets = data_loaders.load_supercam_old(
            DATA_PATH,
            config.DATASET_TYPE
        )
        wvl_mask = config_supercam.WVL_MASKS
        spectral_ranges = config_supercam.SPECTRAL_RANGES
    elif cmd_args.dataset == 'chemcam':
        import config_chemcam
        data, wvl, targets = data_loaders.load_chemcam_old(
            DATA_PATH,
            config.DATASET_TYPE
        )
        wvl_mask = config_chemcam.WVL_MASKS
        spectral_ranges = config_chemcam.SPECTRAL_RANGES
    else:
        raise ValueError(f'Unknown dataset: {cmd_args.dataset}')

    mask_cond = data_preprocessors.mask_wvl(
        wvl,
        wvl_mask
    )
    wvl = wvl[mask_cond]
    data = data.loc[:,mask_cond]


    data = data_preprocessors.normalize_area(
        data,
        wvl,
        spectral_ranges
    )

    if config.NORMALIZE_TO_UNIT_MAXIMUM:
        data = data.apply(
            func=lambda spectrum: pd.Series(
                preprocessing.minmax_scale(spectrum)
            ),
            axis=1
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wvl,
            y=np.squeeze(
                data.mean(axis=0)
            ),
            name=f'Preprocessed {cmd_args.dataset} mean'
        )
    )
    fig = graph_utils._update_layout(fig)
    fig.show()

    targets = targets.loc[data.index]
    targets = targets.loc[
        (targets.loc[:,f'{cmd_args.compound}_Folds'] > 0) & \
          (targets.loc[:,'distance_mm'] < 4000) & \
          np.invert(np.isnan(targets.loc[:,cmd_args.compound])),
        :
    ]

    test_names = targets.loc[
        targets.loc[:,f'{cmd_args.compound}_Folds'] == config.TEST_FOLD,:
    ].index
    train_names = targets.loc[
        targets.loc[:,f'{cmd_args.compound}_Folds'] != config.TEST_FOLD,:
    ].index

    for data_type in ['train', 'test']:
        np.save(
            file=f'/content/{data_type}_targets.npy',
            arr=targets.loc[
                globals().get(f'{data_type}_names'),
                cmd_args.compound
            ].to_numpy()
        )
        np.save(
            file=f'/content/{data_type}_predictors.npy',
            arr=data.loc[
                globals().get(f'{data_type}_names'),
                :
            ].to_numpy()
        )
