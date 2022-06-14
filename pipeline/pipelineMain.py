from loadFunctions import TUH_data,dumpPickles,openPickles
#from braindecode.datasets import create_from_X_y
from clfs import electrodeCLF
import pickle as pickle
from cvFunctions import splitDataset
from datetime import datetime
from os.path import exists
import os, shutil
import json
import pandas as pd
from multiprocessing import freeze_support,set_start_method
from statFunctions import sessionStat
import threading
import numpy as np

if __name__ == '__main__':
    set_start_method("spawn")
    freeze_support()

    # Define path of outer directory for samples:
    path="TUHdata"

    # Create class for data and find all edf files in path, and save in EEG_dict:
    TUH=TUH_data(path=path)

    deletePickle=False
    if exists("TUH_EEG_dict.pkl") and deletePickle:
        os.remove("TUH_EEG_dict.pkl")
    if exists("index_patient_df.pkl") and deletePickle:
        os.remove("index_patient_df.pkl")
    if exists("pickles") and deletePickle:
        for filename in os.listdir("pickles"):
            file_path = os.path.join("pickles", filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    if exists("TUH_EEG_dict.pkl") and exists("index_patient_df.pkl"):
        EEG_dict, index_patient_df = openPickles()
        TUH.EEG_dict = EEG_dict
        TUH.index_patient_df = index_patient_df

        """saved_dict=open("TUH_EEG_dict.pkl","rb")
        TUH.EEG_dict=pickle.load(saved_dict)
        TUH.index_patient_df=pd.read_pickle("index_patient_df.pkl")"""
        print("Preprocessed data loaded succesfully")

    else:
        # Load edf to raw, simple preprocessing, make Xwindows (all windows as arrays) and
        # Ywindows (labels as list of strings) to use for electrode artifact classifier:
        windowssz = 10
        #TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25)
        TUH.parallelElectrodeCLFPrepVer2(tWindow=windowssz, tStep=windowssz * .25)

        # Check i EEG_dict has not already been made. If it has not been made, it means not all preprocessing
        # was succesful and we should instead collect from the pickles:
        if np.sum(TUH.index_patient_df["window_count"].to_list())==0:
            TUH=TUH_data(path="")
            TUH.collectEEG_dictFromPickles()

        dumpPickles(EEG_dict=TUH.EEG_dict, df=TUH.index_patient_df)

        """save_dict=open("TUH_EEG_dict.pkl","wb")
        pickle.dump(TUH.EEG_dict,save_dict)
        save_dict.close()
        TUH.index_patient_df.to_pickle("index_patient_df.pkl")"""
        print("Preprocessed data saved succesfully")

        # plot code begins:
        x = TUH.index_patient_df['patient_id'].tolist()
        y1 = TUH.index_patient_df['elec_count'].tolist()
        y2 = TUH.index_patient_df['window_count'].tolist()
        try:
            y2_m = list()
            for item1, item2 in zip(y2, y1):
                y2_m.append(item1 - item2)
        except:
            y2_m = [0]
            print("Number of recorded counts for elec and windows doesn't match in dataframe")

        plt.bar(x, y1, 0.6, color='r')
        plt.bar(x, y2_m, 0.6, bottom=y1, color='b')
        fig1 = plt.gcf()
        plt.show()
        fig1.savefig("window_and_elec_count.png")

        # Gaussian distribution of elec and window count
        plot = Gaussian.plot(np.mean(y1), np.std(y1), "elec_count")
        plot = Gaussian.plot(np.mean(y2), np.std(y2), "window_count")
        fig2 = plt.gcf()
        plt.show()
        fig2.savefig("Gaussian_window_and_elec_count.png")

        # Plot histogram of window and elec count
        plt.bar(y2, y1, width=2, align='center')  # A bar chart
        fig3 = plt.gcf()
        plt.xlabel('window_count')
        plt.ylabel('elec_count')
        plt.show()
        fig3.savefig("Histogram_window_and_elec_count.png")

    TUH.sessionStat()


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