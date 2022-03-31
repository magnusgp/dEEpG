from loadFunctions import TUH_data
from braindecode.datasets import create_from_X_y

# Define path of outer directory for samples:
path="../TUH_data_sample"
# Define folder for potential saves:
save_dir="D:/fagprojekt"

# Create class for data and find all edf files in path, and save in EEG_dict:
TUH=TUH_data(path=path)

# Load edf to raw, preprocess, make Xwindows (all windows as arrays) and Ywindows (labels as list of strings)
TUH.prep(tWindow=100, tStep=100 * .25)

# Make Braindecode windows dataset from Xwindows and Ywindows:
windows_dataset = create_from_X_y(
    TUH.Xwindows, TUH.Ywindows, drop_last_window=False, sfreq=TUH.sfreq, ch_names=TUH.ch_names,
    window_stride_samples=len(TUH.Xwindows[0][0]),
    window_size_samples=len(TUH.Xwindows[0][0]),
)

windows_dataset.description
