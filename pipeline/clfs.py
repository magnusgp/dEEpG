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
import argparse
import os
import random

def electrodeCLF(TUH, index_df, name = "Nearest Neighbors", Cross_validation = False, Evaluation = False, n_outer_splits = 5, n_inner_splits = 5):
    # Define start time for time measurement:
    start_time = time.time()
    h = 0.02  # step size in the mesh

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
    ]

    if name not in names:
        print("Name not in list of names")
        return

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, verbose=True),
        SVC(gamma=2, C=1, verbose=True),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, verbose=True),
        MLPClassifier(alpha=1, max_iter=1000, verbose=True),
    ]

    #Create dict for classification
    models = dict(zip(names, classifiers))

    if Cross_validation == True:
        n_splits_outer = 5
        n_splits_inner = 5
        print("\n\nInitializing Group Kfold Cross Validation with n = {} outer splits and n = {} inner splits".format(n_splits_outer, n_splits_inner))
        name = name
        mean, std, best_model = finalGroupKFold(name, TUH.index_patient_df, TUH, n_splits_outer=n_splits_outer, n_splits_inner=n_splits_inner, random_state=None)

        # Print the results:
        print("\n\nBest model: {}".format(best_model))
        print("Training best model on all data")

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

    if Evaluation == True:
        pass

    # Create text file with name of the best model, its parameters, its score and the time it took to train it
    with open("results/electrode_clf_results.txt", "a") as f:
        f.write("\n\nBest model: {}".format(best_model))
        f.write("\n\nBest model parameters: {}".format(best_model.get_params()))
        f.write("\n\nBest model score: {} %".format(str(score * 100)))
        f.write("\n\nTime to train model: {}".format(str(time.time() - start_time)))

    return print("Finished processing!")

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', action='store', type=str, required=True)
    parser.add_argument('--outer_splits', action='store', type=int, required=True)
    parser.add_argument('--inner_splits', action='store', type=int, required=True)

    args = parser.parse_args()

    name = args.classifier
    n_outer_splits = args.outer_splits
    n_inner_splits = args.inner_splits

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
    score = electrodeCLF(TUH=TUH, index_df= TUH.index_patient_df, name = name, Cross_validation=True, Evaluation=False, n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits)
    print("Script is done, this is the score:")
    print(score)
