# Prep function fra loadFunctions.py
# Taget fra linje 143
def prep(self, tWindow=100, tStep=100 * .25, plot=False):
    self.tWindow = tWindow
    self.tStep = tStep
    tic = time.time()
    subjects_TUAR19 = defaultdict(dict)
    Xwindows = []
    Ywindows = []
    for k in range(len(self.EEG_dict)):
        subjects_TUAR19[k] = {'path': self.EEG_dict[k]['path']}

        proc_subject = subjects_TUAR19[k]
        proc_subject = self.readRawEdf(proc_subject, tWindow=tWindow, tStep=tStep,
                                       read_raw_edf_param={'preload': True})
        if k == 0 and plot:
            # Plot the energy voltage potential against frequency.
            # proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=128, average=True)

            raw_anno = annotate_TUH(proc_subject["rawData"], annoPath=self.EEG_dict[k]["csvpath"])
            raw_anno.plot()
            plt.show()

        proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
        TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz']  # A1, A2 removed
        proc_subject["rawData"].pick_channels(ch_names=TUH_pick)
        proc_subject["rawData"].reorder_channels(TUH_pick)

        if k == 0 and plot:
            # Plot the energy voltage potential against frequency.
            proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=128, average=True)

            raw_anno = annotate_TUH(proc_subject["rawData"], annoPath=self.EEG_dict[k]["csvpath"])
            raw_anno.plot()
            plt.show()

        preprocessRaw(proc_subject["rawData"], cap_setup="standard_1005", lpfq=1, hpfq=40, notchfq=60,
                      downSam=250)

        if k == 0:

            self.sfreq = proc_subject["rawData"].info["sfreq"]
            self.ch_names = proc_subject["rawData"].info["ch_names"]
            if plot:
                proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=125, average=True)

                raw_anno = annotate_TUH(proc_subject["rawData"], annoPath=self.EEG_dict[k]["csvpath"])
                raw_anno.plot()
                plt.show()

        # Generate output windows for (X,y) as (array, label)
        proc_subject["preprocessing_output"] = slidingRawWindow(proc_subject,
                                                                t_max=proc_subject["rawData"].times[-1],
                                                                tStep=proc_subject["tStep"])

        for window in proc_subject["preprocessing_output"].values():
            Xwindows.append(window[0])
            Ywindows.append(window[1])

    toc = time.time()
    print("\n~~~~~~~~~~~~~~~~~~~~\n"
          "it took %imin:%is to run preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
          "\n~~~~~~~~~~~~~~~~~~~~\n" % (int((toc - tic) / 60), int((toc - tic) % 60), len(subjects_TUAR19),
                                        tWindow, tStep))

    self.Xwindows = Xwindows
    self.Ywindows = Ywindows

# Kode fra clfs.py
    # Error handling for when all labels are the same (due to window size), must be deleted later!
    """
    if len(np.unique(y)) == 1 and y[0][0] == 1:
        y[0] = [0]

    y = np.concatenate([np.array(i) for i in y])

    # Remove first dimension of y
    # Use custom splitting function
    splitDataset(X, y, ratio=0.2, shuffle=True)
    #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    score = {}
    if name == "all":
        # Iterate over all classifiers
        score = {}
        tabdata = []
        start = time.time()
        for name, clf in zip(names, classifiers):
            print("\nNow training: " + name + "\n")
            # Fit classifier
            if not multidim:
                clf.fit(Xtrain, ytrain)
            else:
                clf = MultiOutputClassifier(clf, n_jobs=-1)
                clf.fit(Xtrain, ytrain)
            # Update scoring dictionary
            score[name] = clf.score(Xtest, ytest)
            # Append data to table
            stop = time.time()
            tabdata.append([name, str(round(score[name] * 100, 3)) + " %", str(round(stop - start, 2)) + " s"])
        # Print a formatted table of model performances
        tabdata = sorted(tabdata, key=itemgetter(1), reverse=False)
        print("\n\nModel Performance Summary:")
        print(tabulate(tabdata, headers=['Model name', 'Model score', 'Time'], numalign='left', floatfmt=".3f"))

    elif name in names:
        classifiers[names.index(name)].fit(Xtrain, ytrain)
        score[name] = classifiers[names.index(name)].score(Xtest, ytest)
        print("{} score: {} %".format(name, str(score[name]) * 100))

    else:
        print("Error! Please select a classifier from the list: {}".format(names))
        score = 0.0
    """

    """
    #Find index of best classifier
    best_model = max(score, key=score.get)

    #Match index to classifier name
    for ind, name in enumerate(names):
        if name == best_model:
            best_model_index = ind

    # Save and fit classifier
    new_model = classifiers[best_model_index].fit(Xtrain, ytrain)
    """


    # allgroups ?
    """
    allgroups, X, Y = [], [], []
    # All windows in the same group should have the same group index
    for j in groups.values():
        Xt, Yt, windowInfo = TUH.makeDatasetFromIds(j)
        for k in range(len(j)):
            X.append(Xt[k])
            Y.append(Yt[k])
            for idx in list(groups.values()):
                if len(j) > 1:
                    if j[k] == list(groups.values()).index(idx):
                        for l in range(len(Xt)):
                            if type(l) == int:
                                for __ in range(len(Xt[0])):
                                    allgroups.append(list(groups.keys())[j[k]])
                            else:
                                for __ in range(len(Xt[l])):
                                    allgroups.append(list(groups.keys())[j[k]])
                elif len(j) == 1:
                    if j == list(groups.values()).index(idx):
                        for _ in range(len(Xt[k])):
                            allgroups.append(list(groups.keys())[j[k]])
    """
    ### CLFS.PY
    """import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multioutput import MultiOutputClassifier
from tabulate import tabulate
import pandas as pd
from operator import itemgetter
#import pickle5 as pickle
import pickle
from cvFunctions import CrossValidation_1, CrossValidation_2, splitDataset, GroupKFoldCV, GroupKFold_2, finalGroupKFold
from collections import defaultdict
from tqdm import *
import time
from loadFunctions import TUH_data, openPickles
import argparse
import os
import random

seed = 42
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', action='store', type=int, required=True)
parser.add_argument('--outer splits', action='store', type=int, required=True)

args = parser.parse_args()

artifact = args.artifact
n_epochs = args.epochs

def electrodeCLF(TUH, index_df, name = "all", multidim = True, Cross_validation = False, Evaluation = False, loadFromPickle = False):
    h = 0.02  # step size in the mesh

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
    #    "AdaBoost",
    #    "Naive Bayes",
    #    "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, verbose=True),
        SVC(gamma=2, C=1, verbose=True),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, verbose=True),
        MLPClassifier(alpha=1, max_iter=1000, verbose=True),
        #AdaBoostClassifier(),
        #GaussianNB(),
        #QuadraticDiscriminantAnalysis(),
    ]

    #Create dict for classification
    models = dict(zip(names, classifiers))

    # Check if a saved dataset exists, if not, create it:
    filename = 'TUH.sav'
    # Check if the savedModels folder contains a saved dataset:
    if not os.path.isfile(filename):
        TUH = TUH_data(path=dictpath)
        windowssz = 100
        TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25, plot=False)
        pickle.dump(TUH, open(filename, 'wb'))# Problems with the plots
    else:
        TUH = pickle.load(open(filename, 'rb'))
    # Pickle stuff
    if loadFromPickle:
        TUH = TUH
        windowssz = 100
        #TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25, plot=False)
        #all_ids = TUH.index_patient_df.patient_id.unique()
    else:
        # Non-pickle stuff
        #dictpath = ""
        #TUH = TUH_data(path=dictpath)
        TUH = TUH
        windowssz = 10
        #TUH.parallelElectrodeCLFPrepVer2(tWindow=windowssz, tStep=windowssz * .25)
        #TUH.sessionStat()

    #all_idx = TUH.index_patient_df.index.unique()
    #X, y, windowInfo = TUH.makeDatasetFromIds(ids=all_idx)
    # Only view the first 25 % of the data:
    #for i in range(len(X)):
    #    X[i] = X[i][:int(len(X[i]) * .25)]
    #    y[i] = y[i][:int(len(y[i]) * .25)]

    if Cross_validation == True:
        n_splits = 2
        print("\n\nInitializing Group Kfold Cross Validation with n = {} splits".format(n_splits))
        #C_model_data = CrossValidation_1(models, X, y)
        #C_model_data = GroupKFoldCV(ids = TUH.index_patient_df, X=X, Y=y, models=models, n_splits=n_splits, random_state=42)
        #C_model = C_model_data[0][0]
        #NB_model = models[C_model]
        #best_model = GroupKFold_2(NB_model, C_model, TUH, X, y, TUH.index_patient_df)[2]
        model, name = SVC(C=0.025, kernel='linear', verbose=True), 'Linear SVM'
        mean, std, best_model = finalGroupKFold(name, TUH.index_patient_df, TUH, n_splits_outer=3, n_splits_inner=2, random_state=None)
        # debug mode
        #best_model = GroupKFold_2(SVC(C=0.025, kernel='linear', verbose=True), 'Linear SVM', TUH, X, y, TUH.index_patient_df)[2]
        #best_model = SVC(C=0.001, kernel='linear', verbose=True)

        # Print the results:
        print("\n\nBest model: {}".format(best_model))
        print("Training best model on all data")

        #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
        Xtrain, Xtest, ytrain, ytest = splitDataset(data = TUH.EEG_dict, ratio=0.334, shuffle=True)

        # Print lengths of test and train datasetes in a table
        print("\n\nTrain and Test dataset sizes:")
        print(tabulate([["Train", len(Xtrain)], ["Test", len(Xtest)]], headers=['Dataset', 'Size'], numalign='left'))

        # Fit new best model and summarize findings
        new_model = best_model.fit(Xtrain, ytrain)
        score = new_model.score(Xtest, ytest)
        print("\n\nBest model: {}".format(best_model))
        print("\n\nBest model score: {} %".format(str(score * 100)))

    else:
        pass
        return print("No validation or evalution has been done, due to lack of choice.")

    #Use pickle to save classifier
    #filename = 'finalized_model.sav'
    #pickle.dump(new_model, open(filename, 'wb'))

    if Evaluation == True:
        pass

    return print("Finished processing!")

if __name__ == "__main__":
    pickling = False
    # non pickle stuff
    if not pickling:
        path = "../TUH_data_sample"
        TUH = TUH_data(path=path)
        windowssz = 100
        #TUH.parallelElectrodeCLFPrepVer2(tWindow=windowssz, tStep=windowssz * .25)
        TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25, plot=False)
        TUH.sessionStat()
        P = TUH.index_patient_df
    # pickle stuff
    if pickling:
        path = ""
        TUH = TUH_data(path=path)

        EEG_dict,index_patient_df=openPickles()
        TUH.EEG_dict = EEG_dict
        TUH.index_patient_df = index_patient_df
        TUH.sessionStat()

    # scoring
    score = electrodeCLF(TUH=TUH, index_df= TUH.index_patient_df, name = "all", multidim=False, Cross_validation=True)
    print("Sript is done, this is the score:")
    print(score)
"""