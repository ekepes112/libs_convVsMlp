import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

def load_old_mars_calibration(
    data_path: [Path,str],
) -> tuple[pd.DataFrame,pd.Series,pd.DataFrame]:
    """
    Load the old Mars calibration data from the specified data path.

    Args:
        data_path (Path or str): The path to the directory containing the calibration data. It can be either a string or a Path object.

    Returns:
        data (pd.DataFrame): A pandas DataFrame containing the loaded data.
        wvl (pd.Series): A pandas Series representing the wavelength metadata.
        metadata (pd.DataFrame): A pandas DataFrame containing the metadata, including the composition.
    """
    if isinstance(data_path,str):
        data_path = Path(data_path)
    data = pd.read_csv(
        data_path.joinpath('data.csv'),
        index_col=0,
    )
    wvl = pd.read_csv(
        data_path.joinpath('metadata_wvl.csv'),
        index_col=0,
    )
    wvl = np.squeeze(wvl.to_numpy())
    metadata = pd.read_csv(
        data_path.joinpath('metadata_composition.csv'),
        index_col=0,
    )

    return data,wvl,metadata

def get_kaggle_key_source_path(
    cloud_storage_root_path: str = 'H/My Drive'
) -> Path:
    """
    Returns the source path for the Kaggle API key.

    Args:
        cloud_storage_root_path (str): The root path of the cloud storage (default is 'H/My Drive').

    Returns:
        Path: The source path for the Kaggle API key.
    """
    common_path = 'secrets/kaggle.json'
    if os.name == 'nt':
        kaggle_source_path = Path(cloud_storage_root_path).joinpath(common_path)
    elif os.name == 'posix':
        kaggle_source_path = Path('/content/gdrive/MyDrive').joinpath(common_path)
    return kaggle_source_path

def get_kaggle_key_target_dir() -> Path:
    """
    Returns the path to the Kaggle API key target directory.
    If the target directory does not exist, it is created.

    Args:
        None

    Returns:
        Path: The path to the Kaggle API key target directory.
    """
    if os.name == 'nt':
        kaggle_target_path = Path(f"C:/users/{os.getlogin()}/.kaggle")
    elif os.name == 'posix':
        kaggle_target_path = Path(f"../root/.kaggle")
    if not kaggle_target_path.exists():
        kaggle_target_path.mkdir(parents=True)
    return kaggle_target_path

def copy_kaggle_api_key() -> None:
    """
    Copies the Kaggle API key to the appropriate directory.

    This function copies the Kaggle API key from the source directory to the target directory.

    Args:
        None

    Returns:
        None
    """
    kaggle_target_path = get_kaggle_key_target_dir().joinpath('kaggle.json')
    if not kaggle_target_path.exists():
        shutil.copyfile(
            get_kaggle_key_source_path(),
            get_kaggle_key_target_dir(),
        )
        os.chmod(get_kaggle_key_target_dir(), 600)
    return None

def download_mars_calibration_data(
    data_type: str,
) -> Path:
    """
    Downloads the Mars calibration data based on the specified data type (supercam/chemcam).

    Args:
        data_type (str): The type of data to download.

    Returns:
        Path: The path to the downloaded data.
    """
    data_path = Path(f"./data/{data_type}")
    data_path.mkdir(parents=True, exist_ok=True)
    if data_type == 'supercam':
        data_name = 'supercam-calibration-dataset-20221218'
    data_url = f"erikkepes/{data_name}"
    os.system(f"kaggle datasets download {data_url}")
    shutil.unpack_archive(
        f"{data_name}.zip",
        data_path,
        'zip',
    )
    return data_path
