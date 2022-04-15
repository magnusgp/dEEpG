import mne
from autoreject import AutoReject, get_rejection_threshold, Ransac
from loadFunctions import TUH_data
from raw_utils import oneHotEncoder

def autoReject(XWindows, YWindows, write = False, reject_channels = None):
    #epochs = mne.Epochs(raw, events)

    ar = AutoReject()
    epochs_clean = ar.fit_transform(XWindows)

    epochs_clean.get_annotations_()

    reject = get_rejection_threshold(XWindows)  # get rejection threshold

    return reject

if __name__ == '__main__':
    mnepath = "../TUH_data_sample/131/00013103/s001_2015_09_30/00013103_s001_t000.edf"
    raw = mne.io.read_raw_edf(mnepath, verbose=False)
    #events = mne.find_events(raw, stim_channel=None)
    path = "../TUH_data_sample"
    TUH = TUH_data(path)
    #X, y = TUH.electrodeClassifierPrep(tWindow=100, tStep=100 * .25, plot=False)
    TUH.prep(tWindow=100, tStep=100 * .25, plot=False)
    autoReject(XWindows=TUH.Xwindows)