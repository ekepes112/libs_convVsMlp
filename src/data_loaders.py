import numpy as np
import pandas as pd
from pathlib import Path

def load_chemcam_old(
    data_path:Path,
    dataset_type:str
) -> tuple:
    subfolder = 'chemcam/extended'

    data = pd.read_csv(
        data_path.joinpath(
            f'{subfolder}/{dataset_type}/data.csv'
        ),
        index_col=0
    )

    wvl = pd.read_csv(
        data_path.joinpath(
            f'{subfolder}/{dataset_type}/metadata_wvl.csv'
        ),
        index_col=0
    )

    wvl = np.squeeze(wvl.to_numpy())

    metadata = pd.read_csv(
        data_path.joinpath(
            f'{subfolder}/{dataset_type}/metadata_composition.csv'
        ),
        index_col=0
    )

    return(data,wvl,metadata)


def load_supercam_old(
    data_path:Path,
    dataset_type:str
) -> tuple:
    subfolder = 'supercam'

    data = pd.read_csv(
        data_path.joinpath(
            f'{subfolder}/{dataset_type}/data.csv'
        ),
        index_col=0
    )

    wvl = pd.read_csv(
        data_path.joinpath(
            f'{subfolder}/{dataset_type}/metadata_wvl.csv'
        ),
        index_col=0
    )

    wvl = np.squeeze(wvl.to_numpy())

    metadata = pd.read_csv(
        data_path.joinpath(
            f'{subfolder}/{dataset_type}/metadata_composition.csv'
        ),
        index_col=0
    )

    return(data,wvl,metadata)