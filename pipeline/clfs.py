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
from tabulate import tabulate

def electrodeCLF(X, y, name = "all"):
    h = 0.02  # step size in the mesh

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    score = {}
    if name == "all":
        # Iterate over all classifiers
        score = {}
        tabdata = []
        for name, clf in zip(names, classifiers):
            # Fit classifier
            clf.fit(Xtrain, ytrain)
            # Update scoring dictionary
            score[name] = clf.score(Xtest, ytest)
            # Append data to table
            tabdata.append([name, str(score[name] * 100) + " %"])
        # Print a formatted table of model performances
        print("\n\nModel Performance Summary:")
        print(tabulate(tabdata, headers=['Model name', 'Model score'], numalign='left', floatfmt=".3f"))

    elif name in names:
        classifiers[names.index(name)].fit(Xtrain, ytrain)
        score[name] = classifiers[names.index(name)].score(Xtest, ytest)
        print("{} score: {} %".format(name, str(score[name]) * 100))

    else:
        print("Error! Please select a classifier from the list: {}".format(names))
        score = 0.0

    return score

if __name__ == "__main__":
    X, y = make_classification(n_features=3, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    score = electrodeCLF(X, y, name = "all")
