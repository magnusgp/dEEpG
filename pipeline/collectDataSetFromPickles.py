#Short script created to just use collectDataSetFromPickles.py to collect
# a dataset for debugging use.
from loadFunctions import TUH_data
import pickle

if __name__ == '__main__':
    # Define path of outer directory for samples:
    path = "TUHdata"

    # Create class for data and find all edf files in path, and save in EEG_dict:
    TUH = TUH_data(path=path)

    TUH.collectEEG_dictFromPickles()

    save_dict=open("TUH_EEG_dict.pkl","wb")
    pickle.dump(TUH.EEG_dict,save_dict)
    save_dict.close()
    TUH.index_patient_df.to_pickle("index_patient_df.pkl")
    print("Collected data set from pickles and saved succesfully")