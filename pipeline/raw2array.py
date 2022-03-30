import tempfile

import numpy as np
import matplotlib.pyplot as plt
import mne

import eegProcess

from torch.utils.data import DataLoader, random_split
from torch import Generator

from braindecode.datasets import TUH
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows, scale as multiply)

def raw2array(raw, preproc_params, win_params=None, save_path=None):
    """
    Convert raw EDF data to array.
    """
    # Preprocess
    preproc = Preprocessor(preproc_params)
    preproc.apply(raw)

    # Create windows
    if win_params is not None:
        windows = create_fixed_length_windows(raw, win_params)
    else:
        windows = create_fixed_length_windows(raw)

    # Scale windows
    windows = multiply(windows, preproc_params['scaler'])

    # Save windows
    if save_path is not None:
        np.save(save_path, windows)
    return windows
