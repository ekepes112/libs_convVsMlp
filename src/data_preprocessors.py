import numpy as np
import pandas as pd
import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)

def mask_wvl(wvl:np.array,masks:list):
    keep_mask = np.array([True]*len(wvl))
    for mask in masks:
        keep_mask[
            np.where((wvl >= mask[0]) & (wvl_cc <= mask[1]))
        ] = False
    return(keep_mask)


def normalize_area(
    spectra: pd.DataFrame,
    wvl: np.array,
    ranges: list = []
) -> np.array:
    """
    RETURN a spectrum with unit area <spectrum_proc>:np.array
    """
    log.info(f'spectra shape {spectra.shape}')
    if len(ranges) == 0:
        return(spectra / spectra.sum(axis=0))

    for range in ranges:
      log.debug(f'processing range {range}')
      ndx = np.where((wvl >= range[0]) & (wvl <= range[1]))[0]
      log.debug(f'indices to sum over {ndx}')
      spectra.iloc[:,ndx] = spectra.iloc[:,ndx].divide(
          spectra.iloc[:,ndx].sum(axis=1),
          axis=0
      )

    return(spectra)