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

def select_by_duration(ds, tmin=0, tmax=None):
    if tmax is None:
        tmax = np.inf
    # determine length of the recordings and select based on tmin and tmax
    split_ids = []
    for d_i, d in enumerate(ds.datasets):
        duration = d.raw.n_times / d.raw.info['sfreq']
        if tmin <= duration <= tmax:
            split_ids.append(d_i)
    splits = ds.split(split_ids)
    split = splits['0']
    return split

def select_by_channels(ds, ch_mapping):
    split_ids = []
    for i, d in enumerate(ds.datasets):
        ref = 'ar' if d.raw.ch_names[0].endswith('-REF') else 'le'
        # these are the channels we are looking for
        seta = set(ch_mapping[ref].keys())
        # these are the channels of the recording
        setb = set(d.raw.ch_names)
        # if recording contains all channels we are looking for, include it
        if seta.issubset(setb):
            split_ids.append(i)
    #return ds.split(split_ids)['0']
    return ds.split(split_ids)['0']

def custom_rename_channels(raw, mapping):
    # rename channels which are dependent on referencing:
    # le: EEG 01-LE, ar: EEG 01-REF
    # mne fails if the mapping contains channels as keys that are not present
    # in the raw
    reference = raw.ch_names[0].split('-')[-1].lower()
    assert reference in ['le', 'ref'], 'unexpected referencing'
    reference = 'le' if reference == 'le' else 'ar'
    raw.rename_channels(mapping[reference])


def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    # crop recordings to tmin – tmax. can be incomplete if recording
    # has lower duration than tmax
    # by default mne fails if tmax is bigger than duration
    tmax = min((raw.n_times - 1) / raw.info['sfreq'], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)

def raw2array(raw, preproc_params, win_params, window_size_samples=10 ,window_stride_samples=2, drop_last_window=True, save_path=None, preproc=False, manual_preproc=False):
    """
    Convert raw EDF data to array.
    """
    # Preprocess
    if preproc:
        raw = preprocess(
            concat_ds=raw,
            preprocessors=preproc_params,
            n_jobs=1,
            save_dir=save_path)
    elif manual_preproc:
        for i in range(len(preproc_params)):
            preprocessor = Preprocessor(preproc_params[i])
            #  Apply preprocessing
            preprocessor.apply(raw)

    # Create windows
    if win_params is not None:
        windows = create_fixed_length_windows(raw, win_params)
    else:
        windows = create_fixed_length_windows(raw, window_size_samples=window_size_samples, window_stride_samples=window_stride_samples, drop_last_window=drop_last_window)

    # Save windows
    if save_path is not None:
        np.save(save_path, windows)

    return windows

def oneHotEncoder(labels, enumerate_labels=False, clfbin = False, type = 'labels'):
    """
    Encode labels to one-hot encoding.
    """
    one_hot_labels = []
    if type == 'labels':
        all_labels = ['musc', 'eyem', 'elec', 'eyem_musc', 'musc_elec', 'chew', 'eyem_elev',
                      'eyem_chew', 'shiv', 'chew_musc', 'elpp', 'chew_elec', 'eyem_shiv', 'shiv_elec','null']
    elif type == 'channel':
        all_labels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz']

    n_classes = len(all_labels)
    #labels = [item for sublist in labels for item in sublist]
    for i in range(len(labels)):
        one_hot_labels.append(np.zeros((len(labels[i]), n_classes)))
    for j, label in enumerate(labels):
        for k in range(len(labels[j])):
            if label[k] in all_labels:
                #one_hot_labels[j, all_labels.index(label[k])] = 1
                one_hot_labels[j][k][all_labels.index(label[k])] = 1
                continue

    if enumerate_labels:
        lab = []
        for l in range(len(one_hot_labels)):
            lab2 = []
            for m in range(len(one_hot_labels[l])):
                lab2.append(np.where(one_hot_labels[l][m] == 1)[0].item())
            lab.append(lab2)

        if clfbin:
            labbin = np.zeros(np.shape(lab))
            for n in range(len(lab)):
                labbin[n] = 1 if np.isin(lab[n], [2]).astype(int).any() == 1 else 0
            return labbin

        return lab

    return one_hot_labels

def labelInt(labels):
    all_labels = ['musc', 'eyem', 'elec', 'eyem_musc', 'musc_elec', 'chew', 'eyem_elev',
                  'eyem_chew', 'shiv', 'chew_musc', 'elpp', 'chew_elec', 'eyem_shiv', 'shiv_elec']
    encoded = []
    for i in range(len(labels)):
        for j in range(len(all_labels)):
            if labels[i] == all_labels[j]:
                encoded.append([j+1])
                break
    return encoded