#Short script created to just use collectDataSetFromPickles.py to collect
# a dataset for debugging use.
from loadFunctions import TUH_data
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from loadFunctions import Gaussian
import numpy as np

if __name__ == '__main__':
    # Define empty path so class can be made with no files in it initially:
    path = ""

    # Create class for data and find all edf files in path, and save in EEG_dict:
    TUH = TUH_data(path=path)

    #Comment in this bit if data set is to be collected from pickles:
    TUH.collectEEG_dictFromPickles()
    save_dict = open("TUH_EEG_dict.pkl", "wb")
    pickle.dump(TUH.EEG_dict, save_dict)
    save_dict.close()
    TUH.index_patient_df.to_pickle("index_patient_df.pkl")

    # Comment out this bit if data set is to be collected from pickles:
    # Opens pickles to define data set and info about it
    """
    saved_dict = open("TUH_EEG_dict.pkl", "rb")
    TUH.EEG_dict = pickle.load(saved_dict)
    TUH.index_patient_df = pd.read_pickle("index_patient_df.pkl")
    print("Opened pickle succesfully")"""

    #plot code begins:
    """
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
    fig3.savefig("Histogram_window_and_elec_count.png")"""

