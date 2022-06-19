from loadFunctions import TUH_data,dumpPickles,openPickles
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
import numpy as np
import matplotlib.pyplot as plt
from loadFunctions import Gaussian

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

        x = TUH.index_patient_df['patient_id'].tolist()
        y1 = TUH.index_patient_df['elec_count'].tolist()
        y2 = TUH.index_patient_df['window_count'].tolist()

        try:
            y2_m = list()
            for item1, item2 in zip(y2, y1):
                y2_m.append(item1 - item2)
        except:
            y2_m = [0]
            print("Number of recorded counts for elec and windows dosen't match in dataframe")

        """
        #Nice plot for whole data set with y-axis jump
        f, (ax2, ax) = plt.subplots(2, 1, sharex=True, facecolor='w')
        ax.bar(x, y1, 0.6, color='r', label="elec")
        ax.bar(x, y2_m, 0.6, bottom=y1, color='b', label="null")
        ax2.bar(x, y1, 0.6, color='r', label="elec")
        ax2.bar(x, y2_m, 0.6, bottom=y1, color='b', label="null")

        ax.set_ylim(0, 25000)
        ax2.set_ylim(52000, 71000)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', bottom=False)
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(axis='x',bottom=False)
        ax2.xaxis.set_ticklabels([])

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((1 - d, 1 + d),(-d, +d), **kwargs)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        ax2.legend(loc="upper left")
        plt.ylabel("Window count", size=14)
        f.add_subplot(111,frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel("The data files",size=14)
        plt.title("Count of elec and null windows in data files", size=18)

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig("window_and_elec_count.png", dpi=220)"""



        plt.bar(x, y1, 0.8, color='r',label="elec")
        plt.bar(x, y2_m, 0.8, bottom=y1, color='b', label="null")
        ax1=plt.gca()
        ax1.axes.xaxis.set_ticklabels([])
        plt.legend(loc="upper left")
        plt.ylabel("Window count", size=14)
        plt.xlabel("The data files", size=14)
        plt.title("Elec and null count in data files w. limitation", size=18)
        fig1 = plt.gcf()
        plt.show()
        fig1.savefig("window_and_elec_count.png", dpi=220)


        """# Plot histogram of window and elec countt
        plt.scatter(y2, y1, alpha=0.5, color='black')  # A bar chart
        fig3 = plt.gcf()
        plt.xlabel('Window count',size=14)
        plt.ylabel('Elec count',size=14)
        plt.title("Data files plotted with window vs. elec count",size=18)
        plt.show()
        fig3.savefig("Histogram_window_and_elec_count.png", dpi=220)"""

    else:
        # Load edf to raw, simple preprocessing, make Xwindows (all windows as arrays) and
        # Ywindows (labels as list of strings) to use for electrode artifact classifier:
        windowssz = 5
        #TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25)
        TUH.parallelElectrodeCLFPrepVer2(tWindow=windowssz, tStep=windowssz * .25, limit=1000)

        # Check i EEG_dict has not already been made. If it has not been made, it means not all preprocessing
        # was succesful and we should instead collect from the pickles:
        if np.sum(TUH.index_patient_df["window_count"].to_list())==0:
            TUH=TUH_data(path="")
            TUH.collectEEG_dictFromPickles()

        dumpPickles(EEG_dict=TUH.EEG_dict, df=TUH.index_patient_df)

        print("Preprocessed data saved succesfully")

    #TUH.sessionStat()


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