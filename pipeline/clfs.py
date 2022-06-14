import numpy as np
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
import os

def electrodeCLF(TUH, index_df, name = "all", multidim = True, Cross_validation = False, Evaluation = False):
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
    """
    if not os.path.isfile(filename):
        TUH = TUH_data(path=dictpath)
        windowssz = 100
        TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25, plot=False)
        pickle.dump(TUH, open(filename, 'wb'))# Problems with the plots
    else:
        TUH = pickle.load(open(filename, 'rb'))
    """
    # Pickle stuff
    TUH = TUH
    windowssz = 100
    #TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25, plot=False)
    #all_ids = TUH.index_patient_df.patient_id.unique()

    # Non-pickle stuff
    #TUH = TUH_data(path=dictpath)
    #windowssz = 10
    #TUH.parallelElectrodeCLFPrepVer2(tWindow=windowssz, tStep=windowssz * .25)
    #TUH.sessionStat()

    all_idx = TUH.index_patient_df.index.unique()
    X, y, windowInfo = TUH.makeDatasetFromIds(ids=all_idx)
    # Only view the first 25 % of the data:
    for i in range(len(X)):
        X[i] = X[i][:int(len(X[i]) * .25)]
        y[i] = y[i][:int(len(y[i]) * .25)]

    if Cross_validation == True:
        n_splits = 2
        print("\n\nInitializing Group Kfold Cross Validation with n = {} splits".format(n_splits))
        #C_model_data = CrossValidation_1(models, X, y)
        #C_model_data = GroupKFoldCV(ids = TUH.index_patient_df, X=X, Y=y, models=models, n_splits=n_splits, random_state=42)
        #C_model = C_model_data[0][0]
        #NB_model = models[C_model]
        #best_model = GroupKFold_2(NB_model, C_model, TUH, X, y, TUH.index_patient_df)[2]
        model, name = SVC(C=0.025, kernel='linear', verbose=True), 'Linear SVM'
        mean, std, best_model = finalGroupKFold('Linear SVM', TUH.index_patient_df, X, y)
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

    return print("Model has been evaluated and stored")

if __name__ == "__main__":
    # non pickle stuff
    """
    path = "../TUH_data_sample"
    TUH = TUH_data(path=path)
    windowssz = 10
    TUH.parallelElectrodeCLFPrepVer2(tWindow=windowssz, tStep=windowssz * .25)
    TUH.sessionStat()
    P = TUH.index_patient_df
    """
    # pickle stuff
    path = ""
    TUH = TUH_data(path=path)

    saved_dict = open("TUH_EEG_dict.pkl", "rb")
    TUH.EEG_dict = pickle.load(saved_dict)
    TUH.index_patient_df = pd.read_pickle("index_patient_df.pkl")

    score = electrodeCLF(TUH=TUH, index_df= TUH.index_patient_df, name = "all", multidim=False, Cross_validation=True)
    print("Sript is done, this is the score:")
    print(score)
