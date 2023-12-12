import numpy as np
import pandas as pd
import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)

def mask_wvl(
    wvl:np.array,
    masks:list
) -> np.array:
    """_summary_

    Args:
        wvl (np.array): _description_
        masks (list): _description_

    Returns:
        np.array: _description_

    https://doi.org/10.1016/j.sab.2016.12.003
    """
    keep_mask = np.array([True]*len(wvl))
    for mask in masks:
        keep_mask[
            np.where((wvl >= mask[0]) & (wvl <= mask[1]))
        ] = False
    return keep_mask


def normalize_area(
    spectra: pd.DataFrame,
    wvl: np.array,
    wvl_ranges: list = []
) -> pd.DataFrame:
    """
    RETURN a spectrum with unit area <spectrum_proc>:np.array
    """
    log.info(f'spectra shape {spectra.shape}')
    if len(wvl_ranges) == 0:
        return(spectra / spectra.sum(axis=0))

    for wvl_range in wvl_ranges:
      log.debug(f'processing range {wvl_range}')
      ndx = np.where((wvl >= wvl_range[0]) & (wvl <= wvl_range[1]))[0]
      log.debug(f'indices to sum over {ndx}')
      spectra.iloc[:,ndx] = spectra.iloc[:,ndx].divide(
          spectra.iloc[:,ndx].sum(axis=1),
          axis=0
      )

    return spectra