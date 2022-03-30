import mne

from braindecode.datasets import create_from_X_y
from raw_utils import labelInt
import os, mne, time
import os.path
from mne.io import read_raw_edf
from collections import defaultdict
from datetime import datetime, timezone
import torch, re, warnings
import pandas as pd
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from eegProcess import TUH_rename_ch, nonPipeline, spectrogramMake, slidingWindow, pipeline
from prepareData import TUH_data

path="TUH_data_sample"
save_dir=os.getcwd()

TUH=TUH_data()
TUH.findEdf(path=path)
sfreq = 250
ch_names = sorted([
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'])
#print(TUH.EEG_dict)

TUH.loadAllRaw()
TUH.prep(saveDir=save_dir)

event_codes = labelInt(TUH.Y)

windows_dataset = create_from_X_y(
    TUH.Xraw, event_codes, drop_last_window=False, sfreq=sfreq, ch_names=ch_names,
    window_stride_samples=len(TUH.Xraw[0][0]),
    window_size_samples=len(TUH.Xraw[0][0]),
)

windows_dataset.description  # look as dataset description