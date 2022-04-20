from clfs import electrodeCLF
import mne
#from autoreject import AutoReject, get_rejection_threshold, Ransac
from loadFunctions import TUH_data
from raw_utils import oneHotEncoder, labelInt
from braindecode.datasets import create_from_X_y

if __name__ == "__main__":
    # Create EEG dataset
    path = "../TUH_data_sample"
    TUH = TUH_data(path)

    X, y = TUH.electrodeCLFPrep(tWindow=100, tStep=100 * .25, plot=False)

    y = len(X[0]) * list(map(lambda el:[el.astype(int)], y))

    Xnew = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            Xnew.append(X[i][j])

    score = electrodeCLF(Xnew, y, name = "all", multidim=False)