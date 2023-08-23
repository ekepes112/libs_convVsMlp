import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import plotly.graph_objs as go
import re

import data_loaders
import data_preprocessors
import config
import graph_utils


def get_target_from_spectrum_id(spectrum_id: str):
    return re.sub('^[0-9]{1,}_','',spectrum_id)

def generate_folds(
    concentrations: pd.DataFrame,
    fold_count: int
):
    concentration_bins = pd.cut(
        concentrations,
        bins=100//fold_count,
        right=True,
        labels=None,
        retbins=False,
        precision=3,
        include_lowest=False,
        duplicates='raise',
        ordered=True
    ).astype(str)
    concentration_bins = pd.DataFrame({
        'bin':concentration_bins,
        'target': [
            get_target_from_spectrum_id(x)
            for x
            in concentration_bins.index
        ]
    })
    fold_splitter = StratifiedKFold(
        n_splits=fold_count,
        random_state=97481,
        shuffle=True
    )
    fold_splitter.get_n_splits(
        concentration_bins.loc[~concentration_bins.duplicated(),'target'],
        concentration_bins.loc[~concentration_bins.duplicated(),'bin']
    )

    fold_concentrations = []
    for _, test_index in fold_splitter.split(
        concentration_bins.loc[~concentration_bins.duplicated(),'target'],
        concentration_bins.loc[~concentration_bins.duplicated(),'bin']
    ):
        fold_concentrations.append(
            concentration_bins.loc[
                ~concentration_bins.duplicated(),
                'target'
            ].iloc[test_index]
        )

    return fold_concentrations,concentration_bins


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

    if not f"{cmd_args.compound}_Folds" in list(targets.columns):
        folds, bins = generate_folds(
            concentrations=targets[cmd_args.compound],
            fold_count=len(config.TRAIN_FOLDS) + 1
        )

        for ndx,fold in enumerate(folds):
            targets.loc[
                bins['target'].isin(fold),
                f"{cmd_args.compound}_Folds"
            ] = ndx

        targets[
            f"{cmd_args.compound}_Folds"
        ] = targets[
            f"{cmd_args.compound}_Folds"
        ].astype(np.int8)

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
        targets.loc[
            globals().get(f'{data_type}_names'),
            [cmd_args.compound,f'{cmd_args.compound}_Folds']
        ].to_pickle(f'/content/{data_type}_targets.pkl')
        data.loc[
            globals().get(f'{data_type}_names'),
            :
        ].to_pickle(f'/content/{data_type}_predictors.pkl')
