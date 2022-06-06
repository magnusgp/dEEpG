#Make two level CV for model selection
from numpy import std
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import f1_score
import random
import pandas as pd
from loadFunctions import TUH_data

def CrossValidation_2(model, name, X, Y, n_splits_outer=3, n_splits_inner=2, random_state=None):
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
    cv_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    outer_results = list()
    best_modeL_score = 0

    X = np.squeeze(X)
    Y = np.squeeze(Y)

    for train_index, test_index in cv_outer.split(X):
        #Split the data
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
        # define search space
        space = dict()


        if name == "Nearest Neighbors":
            space['n_neighbors'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if name == "Linear SVM":
            space['C'] = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        if name == "RBF SVM":
            space['gamma'] = [1, 2, 5, 10, 20]
        if name == "Gaussian Process":
            space['kernel'] = ['rbf', 'sigmoid', 'poly']
            space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        if name == "Decision Tree":
            space['max_depth'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if name == "Random Forest":
            space['max_depth'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            space['n_estimators'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            space['max_features'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if name == "Neural Net":
            space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            space['max_itter'] = [1, 10, 100, 1000, 10000]

        # define search
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, Y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(Y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        # store the best performing model
        if acc > best_modeL_score:
            best_modeL_score = acc
            best_model_ = best_model
            best_model_params = result.best_params_
    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), std(outer_results)))
    # report the best configuration
    print('Best Config: %s for model %s' % (best_model_params, best_model_))

    return [np.mean(outer_results), std(outer_results), best_model_]

def CrossValidation_1(models, X, Y, n_splits=3, random_state=None):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results_acc = list()
    results_f1 = list()
    dict = {}
    dict_f1 = {}
    best_model = [[0, 0, 0]]
    for name, model in models.items():
        dict[name] = list()
        dict_f1[name] = list()

    X = np.squeeze(X)
    Y = np.squeeze(Y)

    for train_index, test_index in cv.split(X):
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # evaluate each model in turn
        for name, model in models.items():
            # evaluate the model and store results
            model.fit(X_train, Y_train)
            yhat = model.predict(X_test)
            acc = accuracy_score(Y_test, yhat)
            dict[name].append(acc)
            f1 = f1_score(Y_test, yhat)
            dict_f1[name].append(f1)
            # summarize the results
            print('>%s: %.3f' % (name, acc))
    # summarize the average accuracy
    for name, model in models.items():
        print('%s: %.3f' % (name, np.mean(dict[name])))
        print('%s: %.3f' % (name, np.mean(dict_f1[name])))
        if np.mean(dict[name]) > best_model[0][1]:
            best_model = []
            best_model.append([name, np.mean(dict[name]), np.mean(dict_f1[name])])
    return best_model

def splitDataset(data, ratio, shuffle=False):
    # Function that splits the dataset into test and training based on patient IDs

    # Get patient IDs and shuffle them random
    ids = []
    for i in range(len(data)):
        ids.append(data[i]['patient_id'])

    patients = list(set(ids))
    if shuffle:
        random.shuffle(patients)

    # Make test and training datasets
    test = patients[:int(len(patients) * ratio)]
    train = patients[int(len(patients) * ratio):]

    # If test is empty (we only have 1 data file currently), copy train to test
    # TODO: This is very bad practice and should be fixed immediately
    if len(test) == 0:
        test = train

    # Make test and training datasets
    test_data = []
    train_data = []
    for i in range(len(data)):
        if data[i]['patient_id'] in test:
            test_data.append(data[i])
        if data[i]['patient_id'] in train:
        # TODO: should be elif so we can save checks
        # elif data[i]['patient_id'] in train:
            train_data.append(data[i])

    X_test, Y_test, X_train, Y_train = [], [], [], []

    # Split test and training data into X and Y
    for i in range(len(test_data)):
        for j in range(len(test_data[i]['labeled_windows'])):
            X_test.append([test_data[0]['labeled_windows'][k] for k in test_data[0]['labeled_windows'].keys()][j][0])
            Y_test.append([test_data[0]['labeled_windows'][k] for k in test_data[0]['labeled_windows'].keys()][j][1])

    for i in range(len(train_data)):
        for j in range(len(train_data[i]['labeled_windows'])):
            X_train.append([train_data[0]['labeled_windows'][k] for k in train_data[0]['labeled_windows'].keys()][j][0])
            Y_train.append([train_data[0]['labeled_windows'][k] for k in train_data[0]['labeled_windows'].keys()][j][1])

    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":

    Xtrain, Xtest, ytrain, ytest = splitDataset(TUH.index_patient_df, ratio=0.2, shuffle=True)