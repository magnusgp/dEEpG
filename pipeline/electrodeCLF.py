from clfs import electrodeCLF
import mne
#from autoreject import AutoReject, get_rejection_threshold, Ransac
from loadFunctions import TUH_data
from raw_utils import oneHotEncoder, labelInt
from braindecode.datasets import create_from_X_y

# Create EEG dataset
path = "../TUH_data_sample"
TUH = TUH_data(path)
#Xwindows, Ywindows = TUH.electrodeClassifierPrep(tWindow=100, tStep=100 * .25, plot=False)
TUH.prep(tWindow=100, tStep=100 * .25, plot=False)

windows_dataset = create_from_X_y(
    TUH.Xwindows, TUH.Ywindows, drop_last_window=False, sfreq=TUH.sfreq, ch_names=TUH.ch_names,
    window_stride_samples=len(TUH.Xwindows[0][0]),
    window_size_samples=len(TUH.Xwindows[0][0]),)

i = 0
x_i, y_i, window_ind = windows_dataset[0]
n_channels, n_times = x_i.shape  # the EEG data
_, start_ind, stop_ind = window_ind
print(f"n_channels={n_channels}  -- n_times={n_times} -- y_i={y_i}")
print(f"start_ind={start_ind} -- stop_ind={stop_ind}")

print(windows_dataset.description)

y = TUH.Ywindows
X = TUH.Xwindows

y = len(X[0]) * y
X2 = []
Xnew = []
for i in range(len(X)):
    for j in range(len(X[0])):
        Xnew.append(X[i][j])
    #X2.append(Xnew)

#score = electrodeCLF(Xnew, oneHotEncoder(y, enumerate_labels=False))
score = electrodeCLF(Xnew, oneHotEncoder(y, enumerate_labels=True), name = "all")