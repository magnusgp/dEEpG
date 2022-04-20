from loadFunctions import TUH_data
from braindecode.datasets import create_from_X_y
from clfs import electrodeCLF

# Define path of outer directory for samples:
path="../TUH_data_sample"
# Define folder for potential saves:
save_dir="D:/fagprojekt"

# Create class for data and find all edf files in path, and save in EEG_dict:
TUH=TUH_data(path=path)

# Load edf to raw, simple preprocessing, make Xwindows (all windows as arrays) and
# Ywindows (labels as list of strings) to use for electrode artifact classifier:
windowssz = 10
elecX,elecY=TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25, plot=False)
# Find the best electrode artifact classifier:
electrodeCLF(elecX, elecY, "all", False)

# Load edf to raw, full preprocess with electrode classifier, make Xwindows (all windows
# as arrays) and Ywindows (labels as list of strings) to use for data augmentation.
TUH.prep(tWindow=100, tStep=100 * .25,plot=True)

# Make Braindecode windows dataset from Xwindows and Ywindows:
windows_dataset = create_from_X_y(
    TUH.Xwindows, TUH.Ywindows, drop_last_window=False, sfreq=TUH.sfreq, ch_names=TUH.ch_names,
    window_stride_samples=len(TUH.Xwindows[0][0]),
    window_size_samples=len(TUH.Xwindows[0][0]),
)

windows_dataset.description
