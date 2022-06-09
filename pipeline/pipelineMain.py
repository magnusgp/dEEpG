from loadFunctions import TUH_data
#from braindecode.datasets import create_from_X_y
from clfs import electrodeCLF
import pickle
from cvFunctions import splitDataset
from datetime import datetime
from os.path import exists
import os, shutil
import json
import pandas as pd
from multiprocessing import freeze_support,set_start_method
from statFunctions import sessionStat
import threading


if __name__ == '__main__':
    set_start_method("spawn")
    freeze_support()

    # Define path of outer directory for samples:
    path="TUHdata"

    # Create class for data and find all edf files in path, and save in EEG_dict:
    TUH=TUH_data(path=path)

    deletePickle=True
    if exists("TUH_EEG_dict.pkl") and deletePickle:
        os.remove("TUH_EEG_dict.pkl")
    if exists("index_patient_df.pkl") and deletePickle:
        os.remove("index_patient_df.pkl")
    if exists("pickles"):
        for filename in os.listdir("pickles"):
            file_path = os.path.join("pickles", filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    if exists("TUH_EEG_dict.pkl") and exists("index_patient_df.pkl"):
        saved_dict=open("TUH_EEG_dict.pkl","rb")
        TUH.EEG_dict=pickle.load(saved_dict)
        TUH.index_patient_df=pd.read_pickle("index_patient_df.pkl")
        print("Preprocessed data loaded succesfully")

    else:
        # Load edf to raw, simple preprocessing, make Xwindows (all windows as arrays) and
        # Ywindows (labels as list of strings) to use for electrode artifact classifier:
        windowssz = 10
        #TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25)
        TUH.parallelElectrodeCLFPrepVer2(tWindow=windowssz, tStep=windowssz * .25)

        save_dict=open("TUH_EEG_dict.pkl","wb")
        pickle.dump(TUH.EEG_dict,save_dict)
        save_dict.close()
        TUH.index_patient_df.to_pickle("index_patient_df.pkl")
        print("Preprocessed data saved succesfully")

    #sessionStat(EEG_dict=TUH.EEG_dict)

"""
elecX,elecY,windowInfo=TUH.makeDatasetFromIds(ids=[0])

# Save class instance to pickle for later loading
#pickle.dump(TUH)

Xtrain, Xtest, ytrain, ytest = splitDataset(TUH.index_patient_df, ratio=0.2, shuffle=True)

# Find the best electrode artifact classifier:
bestmodel=electrodeCLF(elecX, elecY, "all", False)

#bads=classifyElectrodeIntervals(elecX,windowInfo,bestmodel)


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
"""