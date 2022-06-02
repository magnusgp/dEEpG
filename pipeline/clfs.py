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
import pickle
from cvFunctions import CrossValidation_1, CrossValidation_2
from collections import defaultdict
from tqdm import *
import time
from loadFunctions import TUH_data

def electrodeCLF(dictpath, name = "all", multidim = True, Cross_validation = False):
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
    models = zip(names, classifiers)

    TUH = TUH_data(path=dictpath)
    windowssz = 100
    TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25, plot=False)  # Problems with the plots
    all_ids = TUH.index_patient_df.patient_id.unique()
    all_idx = TUH.index_patient_df.index.unique()
    x, y, windowInfo = TUH.makeDatasetFromIds(ids=all_idx)

    # Error handling for when all labels are the same (due to window size), must be deleted later!
    """
    if len(np.unique(y)) == 1 and y[0][0] == 1:
        y[0] = [0]

    y = np.concatenate([np.array(i) for i in y])
    """
    # Remove first dimension of y
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
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

    if Cross_validation:
        C_model_data = CrossValidation_1(models, X, y)
        C_model = C_model_data[0][0]
        NB_model = models[C_model]
        best_model = CrossValidation_2(NB_model, C_model, X, y)[2]

        new_model = best_model.fit(Xtrain, ytrain)

    else:
        #Find index of best classifier
        best_model = max(score, key=score.get)

        #Match index to classifier name
        for ind, name in enumerate(names):
            if name == best_model:
                best_model_index = ind

        # Save and fit classifier
        new_model = classifiers[best_model_index].fit(Xtrain, ytrain)


    #Use pickle to save classifier
    filename = 'finalized_model.sav'
    pickle.dump(new_model, open(filename, 'wb'))

    return score

if __name__ == "__main__":
    X, y = make_classification(n_features=3, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    path = "../TUH_data_sample"
    score = electrodeCLF(dictpath=path, name = "all", multidim=False, Cross_validation=True)
