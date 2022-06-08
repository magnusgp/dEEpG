#Make two level CV for model selection
from numpy import std
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
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
    outer_results_acc = list()
    outer_results_f1 = list()
    outer_results_BA = list()
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
        # We can do more jobs here, check documentation
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, Y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(Y_test, yhat)
        f1 = f1_score(Y_test, yhat)
        BA = balanced_accuracy_score(Y_test, yhat)
        # store the result
        outer_results_acc.append(acc)
        outer_results_f1.append(f1)
        outer_results_BA.append(BA)
        # report progress
        print('>acc=%.3f, f1_score=%.3f, b_acc_score=%.3f, est=%.3f, cfg=%s' % (acc, f1, BA, result.best_score_, result.best_params_))
        # store the best performing model
        if acc > best_modeL_score:
            best_modeL_score = acc
            best_model_ = best_model
            best_model_params = result.best_params_
    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results_acc), std(outer_results_acc)))
    print('f1 score: %.3f (%.3f)' % (np.mean(outer_results_f1), std(outer_results_f1)))
    print('Balanced accuracy: %.3f (%.3f)' % (np.mean(outer_results_BA), std(outer_results_BA)))
    # report the best configuration
    print('Best Config based in acc: %s for model %s' % (best_model_params, best_model_))

    return [np.mean(outer_results), std(outer_results), best_model_]

def CrossValidation_1(models, X, Y, n_splits=3, random_state=None):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results_acc = list()
    results_f1 = list()
    dict = {}
    dict_f1 = {}
    dict_BA = {}
    best_model = [[0, 0, 0]]
    for name, model in models.items():
        dict[name] = list()
        dict_f1[name] = list()
        dict_BA[name] = list()

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
            BA = balanced_accuracy_score(Y_test, yhat)
            dict_BA[name].append(BA)
            # summarize the results
            print('>%s: %.3f' % (name, acc))
            print('>%s: %.3f' % (name, f1))
            print('>%s: %.3f' % (name, BA))
    # summarize the average accuracy
    for name, model in models.items():
        print('%s: %.3f' % (name, np.mean(dict[name])))
        print('%s: %.3f' % (name, np.mean(dict_f1[name])))
        print('%s: %.3f' % (name, np.mean(dict_BA[name])))
        if np.mean(dict[name]) > best_model[0][1]:
            best_model = []
            best_model.append([name, np.mean(dict[name]), np.mean(dict_f1[name])])
    return best_model

def splitDataset(data, ratio, shuffle=False, Kfold = False):
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
            X_test.append([test_data[i]['labeled_windows'][k] for k in test_data[i]['labeled_windows'].keys()][j][0])
            Y_test.append([test_data[i]['labeled_windows'][k] for k in test_data[i]['labeled_windows'].keys()][j][1])

    for i in range(len(train_data)):
        for j in range(len(train_data[i]['labeled_windows'])):
            X_train.append([train_data[i]['labeled_windows'][k] for k in train_data[i]['labeled_windows'].keys()][j][0])
            Y_train.append([train_data[i]['labeled_windows'][k] for k in train_data[i]['labeled_windows'].keys()][j][1])

    return X_train, X_test, Y_train, Y_test

def GroupKFoldCV(ids, X, Y, models, n_splits=5, random_state=None):
    # Function that splits the dataset into groups for KFold cross validation

    # Group data by unique patient ID
    groups = []
    for i in range(len(ids)):
        groups.append(ids['patient_id'][i])
        for _ in range(len(X)):
            # All windows in the same group should have the same index
            groups[i] = [groups[i]] * len(X[i])

    print("Total number of groups found: ", len(groups))

    # Each group should consist of all session from one patient
    groups = np.squeeze(groups)

    X = np.squeeze(X)
    Y = np.squeeze(Y)

    # TODO: Fix this when more groups has been added from the data
    #gidx = len(groups)//2

    #groups[gidx:] = ['00013202'] * len(groups[gidx:])

    group_kfold = GroupKFold(n_splits=len(groups))
    group_kfold.get_n_splits(X, Y, groups)

    results_acc = list()
    results_f1 = list()
    dict = {}
    dict_f1 = {}
    best_model = [[0, 0, 0]]
    for name, model in models.items():
        dict[name] = list()
        dict_f1[name] = list()
    print("Starting KFold CV with n = %d folds" % group_kfold.n_splits)

    for train_index, test_index in group_kfold.split(X, Y, groups):
        print("TRAIN:", train_index, "TEST:", test_index)

        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # evaluate each model in turn
        for name, model in models.items():
            # evaluate the model and store results
            # TODO: Bug here?
            model.fit(X_train[0], Y_train[0])
            yhat = model.predict(X_test[0])
            acc = accuracy_score(Y_test[0], yhat)
            dict[name].append(acc)
            f1 = f1_score(Y_test[0], yhat)
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

def GroupKFold_2(model, name, TUH, X, Y, ids, n_splits_outer=3, n_splits_inner=2, random_state=None):
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

    # Group data by unique patient ID
    groups = []
    for i in range(len(ids)):
        groups.append(ids['patient_id'][i])
        for _ in range(len(X)):
            # All windows in the same group should have the same index
            groups[i] = [groups[i]] * len(X[i])

    # Each group should consist of all session from one patient
    groups = np.squeeze(groups)

    ids = [TUH.EEG_dict[id]['id'] for id in range(len(TUH.EEG_dict))]

    X = np.squeeze(X)
    Y = np.squeeze(Y)

    #X, Y, _ = TUH.makeDatasetFromIds(ids=ids)

    # TODO: Fix this when more groups has been added from the data
    #gidx = len(groups)//2

    #groups[gidx:] = ['00013202'] * len(groups[gidx:])

    group_kfold_outer = GroupKFold(n_splits=len(groups))
    group_kfold_outer.get_n_splits(X, Y, groups)

    # cv_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    outer_results = list()
    best_modeL_score = 0

    X = np.squeeze(X)
    Y = np.squeeze(Y)

    for train_index, test_index in group_kfold_outer.split(X, Y, groups):
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        g_train, g_test = groups[train_index], groups[test_index]

        # configure the cross-validation procedure
        # cv_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
        group_kfold_inner = GroupKFold(n_splits=len(groups))
        group_kfold_inner.get_n_splits(X, Y, groups)
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
        # We can do more jobs here, check documentation
        search = GridSearchCV(model, space, scoring='accuracy', cv=group_kfold_inner, refit=True)
        # execute search
        # TODO: Endnu en røver pga datamangel
        #result = search.fit(X_train, Y_train, groups=g_train)
        #g_train1 = g_train[:len(g_train)//2]
        #g_train2 = g_train[len(g_train)//2:]
        g_train = [x for xs in g_train for x in xs]
        g_test = [x for xs in g_test for x in xs]
        result = search.fit(X_train[0], Y_train[0], groups=g_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test[0], groups=g_test)
        # evaluate the model
        acc = accuracy_score(Y_test[0], yhat)
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

if __name__ == "__main__":
    pass
    #Xtrain, Xtest, ytrain, ytest = splitDataset(TUH.index_patient_df, ratio=0.2, shuffle=True)