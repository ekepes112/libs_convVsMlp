def import_chemcam(dataset_type):
    subfolder = 'chemcam/extended'

    data = pd.read_csv(
      DATA_PATH.joinpath(
          f'{subfolder}/{dataset_type}/data.csv'
      ),
      index_col=0
    )

    wvl = pd.read_csv(
      DATA_PATH.joinpath(
          f'{subfolder}/{dataset_type}/metadata_wvl.csv'
      ),
      index_col=0
    )

    wvl = np.squeeze(wvl.to_numpy())

    metadata = pd.read_csv(
      DATA_PATH.joinpath(
          f'{subfolder}/{dataset_type}/metadata_composition.csv'
      ),
      index_col=0
    )

    return(data,wvl,metadata)