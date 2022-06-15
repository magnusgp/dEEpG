#Make two level CV for model selection
from collections import defaultdict
from numpy import std
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

# classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# score metric
from sklearn.metrics import f1_score
import random
import pandas as pd
from loadFunctions import TUH_data
import matplotlib.pyplot as plt

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
            best_model.append([name, np.mean(dict[name]), np.mean(dict_f1[name]), np.mean(dict_BA[name])])
    return best_model

def splitDataset(data, ratio, shuffle=False):
    # Function that splits the dataset into test and training based on patient IDs

    # Get patient IDs and shuffle them random
    ids = []
    for i in data.keys():
        ids.append(data[i]['patient_id'])

    patients = list(set(ids))
    if shuffle:
        random.shuffle(patients)

    # Make test and training datasets
    test = patients[:int(len(patients) * ratio)]
    train = patients[int(len(patients) * ratio):]

    # If test is empty raise throw eexception
    if len(test) == 0:
        raise Exception("Test set is empty")

    # Make test and training datasets
    test_data = []
    train_data = []
    for i in data.keys():
        if data[i]['patient_id'] in test:
            test_data.append(data[i])
        elif data[i]['patient_id'] in train:
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
    """
    # Group data by unique patient ID
    groups = []
    for i in range(len(ids)):
        groups.append(ids['patient_id'][i])
        for _ in range(len(X)):
            # All windows in the same group should have the same group index
            groups[i] = [groups[i]] * len(X[i])
    """
    # Use a defauldict to group windows by patient ID
    groups = defaultdict(list)
    for i in range(len(ids)):
        groups[ids['patient_id'][i]].append(i)

    allgroups = []

    # All windows in the same group should have the same group index
    for j in groups.values():
        Xt, Yt = TUH.makeDatasetFromIds(j)
        for _ in range(len(Xt)):
            allgroups.append(list(groups.keys())[j])
        X.append(Xt)
        Y.append(Yt)

    # Merge windows lists in X that have the same group index
    for i in range(len(X)):
        for j in range(len(groups.values())):
            if i in list(groups.values())[j]:
                if len(list(groups.values())[j]) > 1 and i != j:
                    X[i] = X[i] + X[j]
                    X.pop(j)
                    Y[i] = Y[i] + Y[j]
                    Y.pop(j)

    print("\n\nTotal number of groups found: ", len(groups))
    print("\n\nTotal length of all groups: ", len(allgroups))

    # Split the allgroups into n sublists by ID
    groupsset = list(set(list(allgroups)))
    groupsset = [groupsset[i] for i in range(groupsset.count(i))]

    X = np.squeeze(X)
    X = [x for xs in X for x in xs]
    Y = np.squeeze(Y)
    Y = [x for xs in Y for x in xs]

    # TODO: Fix this when more groups has been added from the data
    #gidx = len(groups)//2

    #groups[gidx:] = ['00013202'] * len(groups[gidx:])

    group_kfold = GroupKFold(n_splits=len(groups))
    group_kfold.get_n_splits(X, Y, groups)

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



    print("Starting KFold CV with n = %d folds" % group_kfold.n_splits)

    for train_index, test_index in group_kfold.split(X, Y, allgroups):
        print("TRAIN:", train_index, "TEST:", test_index)

        # Split the data
        X_train, X_test = list(map(X.__getitem__, train_index)), list(map(X.__getitem__, test_index))
        Y_train, Y_test = list(map(Y.__getitem__, train_index)), list(map(Y.__getitem__, test_index))
        #X_train, X_test = X[train_index], X[test_index]
        #Y_train, Y_test = Y[train_index], Y[test_index]
        # evaluate each model in turn
        for name, model in models.items():
            # evaluate the model and store results
            # TODO: Bug here?
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
            best_model.append([name, np.mean(dict[name]), np.mean(dict_f1[name]), np.mean(dict_BA[name])])
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
    #groups = []
    #for i in range(len(ids)):
    #    groups.append(ids['patient_id'][i])

        #for _ in range(len(X)):
        #    # All windows in the same group should have the same index
        #    groups[i] = [groups[i]] * len(X[i])

    groups = defaultdict(list)
    for i in range(len(ids)):
        groups[ids['patient_id'][i]].append(i)

    allgroups = []

    # All windows in the same group should have the same group index
    for j in groups.values():
        Xt, Yt = TUH.makeDatasetFromIds(j)
        for _ in range(len(Xt)):
            allgroups.append(list(groups.keys())[j])
        X.append(Xt)
        Y.append(Yt)

    # Merge windows lists in X that have the same group index
    """
    for i in range(len(X)):
        for j in range(len(groups.values())):
            if i in list(groups.values())[j]:
                if len(list(groups.values())[j]) > 1 and i != j:
                    X[i] = X[i] + X[j]
                    X.pop(j)
                    Y[i] = Y[i] + Y[j]
                    Y.pop(j)
    """

    print("\n\nTotal number of groups found: ", len(groups))
    print("\n\nTotal length of all groups: ", len(allgroups))

    X = np.squeeze(X)
    X = [x for xs in X for x in xs]
    Y = np.squeeze(Y)
    Y = [x for xs in Y for x in xs]

    # Each group should consist of all session from one patient
    #groups = np.squeeze(groups)

    #X, Y, _ = TUH.makeDatasetFromIds(ids=ids)

    # TODO: Fix this when more groups has been added from the data
    #gidx = len(groups)//2

    #groups[gidx:] = ['00013202'] * len(groups[gidx:])

    group_kfold_outer = GroupKFold(n_splits=len(groups))
    group_kfold_outer.get_n_splits(X, Y, groups)

    # cv_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    outer_results = list()
    outer_results_f1 = list()
    outer_results_BA = list()
    best_modeL_score = 0

    for train_index, test_index in group_kfold_outer.split(X, Y, allgroups):
        # Split the data
        X_train, X_test = list(map(X.__getitem__, train_index)), list(map(X.__getitem__, test_index))
        Y_train, Y_test = list(map(Y.__getitem__, train_index)), list(map(Y.__getitem__, test_index))
        #X_train, X_test = X[train_index], X[test_index]
        #Y_train, Y_test = Y[train_index], Y[test_index]
        #g_train, g_test = groups[train_index], groups[test_index]

        # configure the cross-validation procedure
        # cv_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
        group_kfold_inner = GroupKFold(n_splits=len(list(set(map(allgroups.__getitem__, train_index)))))
        group_kfold_inner.get_n_splits(X_train, Y_train, list(map(allgroups.__getitem__, train_index)))

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
        print("\n\nTraining model: ", name)

        result = search.fit(X_train, Y_train, groups=list(map(allgroups.__getitem__, train_index)))
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(Y_test, yhat)
        f1 = f1_score(Y_test, yhat)
        BA = balanced_accuracy_score(Y_test, yhat)
        # store the result
        outer_results.append(acc)
        outer_results_f1.append(f1)
        outer_results_BA.append(BA)
        # report progress
        print('>acc=%.3f, f1_score=%.3f, b_acc_score=%.3f, est=%.3f, cfg=%s' % (
            acc, f1, BA, result.best_score_, result.best_params_))
        # store the best performing model
        if acc > best_modeL_score:
            best_modeL_score = acc
            best_model_ = best_model
            best_model_params = result.best_params_
    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), std(outer_results)))
    print('f1 score: %.3f (%.3f)' % (np.mean(outer_results_f1), std(outer_results_f1)))
    print('Balanced accuracy: %.3f (%.3f)' % (np.mean(outer_results_BA), std(outer_results_BA)))
    # report the best configuration
    print('Best Config based in acc: %s for model %s' % (best_model_params, best_model_))

    return [np.mean(outer_results), std(outer_results), best_model_]

def finalGroupKFold(name, ids, TUH, n_splits_outer=3, n_splits_inner=2, random_state=None):
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
    if name not in names:
        print("The selected model is not valid. Please try again mark")

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


    groups = defaultdict(list)
    for i in ids.index:
        groups[ids['patient_id'][i]].append(ids['index'][i])

    allgroups, X, Y = [], [], []
    # All windows in the same group should have the same group index
    for j in groups.values():
        Xt, Yt, windowInfo = TUH.makeDatasetFromIds(j)
        c = 0
        for k in range(len(j)):
            X.append(Xt[k])
            Y.append(Yt[k])
            c += len(Xt[k])
        allgroups.append([c * [list(groups.keys())[list(groups.values()).index(j)]]])


    allgroups = [g for gs in allgroups for g in gs]
    allgroups = [g for gs in allgroups for g in gs]



    #X = np.squeeze(X)
    X = [x for xs in X for x in xs]
    #Y = np.squeeze(Y)
    Y = [y for ys in Y for y in ys]

    group_kfold_outer = GroupKFold(n_splits=len(groups))
    group_kfold_outer.get_n_splits(X, Y, groups)

    outer_results = list()
    outer_results_f1 = list()
    outer_results_BA = list()
    best_modeL_score = 0

    split_train_plot = []
    split_train_plot_elec = []
    split_train_plot_F = []
    split_train_elec_F = []
    split_test_plot = []
    split_test_plot_elec = []
    split_test_plot_F = []
    split_test_elec_F = []

    male_count_train = 0
    female_count_train = 0
    unknown_count_train = 0
    male_count_train_F = []
    female_count_train_F = []
    unknown_count_train_F = []

    male_count_test = 0
    female_count_test = 0
    unknown_count_test = 0
    male_count_test_F = []
    female_count_test_F = []
    unknown_count_test_F = []

    age_train = []
    age_train_F = []
    age_test = []
    age_test_F = []

    for train_index, test_index in group_kfold_outer.split(X, Y, allgroups):
        # Split the data

        for k in list(set(map(allgroups.__getitem__, train_index))):
            if 'Male' in ids[ids['patient_id']==k]['Gender'].tolist():
                male_count_train += 1
            elif 'Female' in ids[ids['patient_id']==k]['Gender'].tolist():
                female_count_train += 1
            else:
                unknown_count_train += 1
            split_train_plot.append(sum(ids[ids['patient_id']==k]['window_count'].tolist()))
            split_train_plot_elec.append(sum(ids[ids['patient_id'] == k]['elec_count'].tolist()))

            age_train.append(ids[ids['patient_id']==k]['Age'].tolist()[0])

        age_train_F.append(age_train)
        age_train = []

        split_train_plot_F.append(sum(split_train_plot))
        split_train_elec_F.append(sum(split_train_plot_elec))
        split_train_plot = []
        split_train_plot_elec = []

        male_count_train_F.append(male_count_train)
        female_count_train_F.append(female_count_train)
        unknown_count_train_F.append(unknown_count_train)
        male_count_train = 0
        female_count_train = 0
        unknown_count_train = 0

        for k in list(set(map(allgroups.__getitem__, test_index))):
            if 'Male' in ids[ids['patient_id']==k]['Gender'].tolist():
                male_count_test += 1
            elif 'Female' in ids[ids['patient_id']==k]['Gender'].tolist():
                female_count_test += 1
            else:
                unknown_count_test += 1

            split_test_plot.append(sum(ids[ids['patient_id'] == k]['window_count'].tolist()))
            split_test_plot_elec.append(sum(ids[ids['patient_id'] == k]['elec_count'].tolist()))

            age_test.append(ids[ids['patient_id'] == k]['Age'].tolist()[0])
        age_test_F.append(age_test)
        age_test = []

        split_test_plot_F.append(sum(split_test_plot))
        split_test_elec_F.append(sum(split_test_plot_elec))
        split_test_plot = []
        split_test_plot_elec = []

        male_count_test_F.append(male_count_test)
        female_count_test_F.append(female_count_test)
        unknown_count_test_F.append(unknown_count_test)
        male_count_test = 0
        female_count_test = 0
        unknown_count_test = 0

        X_train, X_test = list(map(X.__getitem__, train_index)), list(map(X.__getitem__, test_index))
        Y_train, Y_test = list(map(Y.__getitem__, train_index)), list(map(Y.__getitem__, test_index))

        # configure the cross-validation procedure
        group_kfold_inner = GroupKFold(n_splits=len(list(set(map(allgroups.__getitem__, train_index)))))
        group_kfold_inner.get_n_splits(X_train, Y_train, list(map(allgroups.__getitem__, train_index)))

        space = dict()

        if name == "Nearest Neighbors":
            space['n_neighbors'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            model = classifiers[0]
        if name == "Linear SVM":
            space['C'] = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            model = classifiers[1]
        if name == "RBF SVM":
            space['gamma'] = [1, 2, 5, 10, 20]
            model = classifiers[2]
        if name == "Gaussian Process":
            space['kernel'] = ['rbf', 'sigmoid', 'poly']
            space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            model = classifiers[3]
        if name == "Decision Tree":
            space['max_depth'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            model = classifiers[4]
        if name == "Random Forest":
            space['max_depth'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            space['n_estimators'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            space['max_features'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            model = classifiers[5]
        if name == "Neural Net":
            space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            space['max_itter'] = [1, 10, 100, 1000, 10000]
            model = classifiers[6]

        # define search
        # We can do more jobs here, check documentation
        search = GridSearchCV(model, space, scoring='accuracy', cv=group_kfold_inner, refit=True, n_jobs=-1)
        # execute search
        print("\n\nTraining model: ", name)

        result = search.fit(X_train, Y_train, groups=list(map(allgroups.__getitem__, train_index)))
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(Y_test, yhat)
        f1 = f1_score(Y_test, yhat)
        BA = balanced_accuracy_score(Y_test, yhat)
        # store the result
        outer_results.append(acc)
        outer_results_f1.append(f1)
        outer_results_BA.append(BA)
        # report progress
        print('>acc=%.3f, f1_score=%.3f, b_acc_score=%.3f, est=%.3f, cfg=%s' % (
            acc, f1, BA, result.best_score_, result.best_params_))
        # store the best performing model
        if acc > best_modeL_score:
            best_modeL_score = acc
            best_model_ = best_model
            best_model_params = result.best_params_

    # prop for split
    Prop_train = []
    E_mes_train = np.ones(len(split_train_elec_F))
    k = np.zeros(len(split_train_elec_F))
    for i in range(len(split_train_elec_F)):
        k = split_train_elec_F[i] / split_train_plot_F[i]
        E_mes_train[i] = E_mes_train[i] - k
        Prop_train.append(k)
    E_mes_train = list(E_mes_train)


    Prop_test = []
    E_mes_test = np.ones(len(split_test_elec_F))
    u = np.zeros(len(split_test_elec_F))
    for i in range(len(split_test_elec_F)):
        u = split_test_elec_F[i] / split_test_plot_F[i]
        E_mes_test[i] = E_mes_test[i] - u
        Prop_test.append(u)
    E_mes_test = list(E_mes_test)


    #Plot distribution of elec and window count in splits (bars)
    x = list(np.arange(1, len(Prop_train)+1))
    #Plot results - train
    plt.bar(x, Prop_train, 0.6, color='r', label = "elec")
    plt.bar(x, E_mes_train, 0.6, bottom=Prop_train, color='b', label="null")
    plt.legend(loc = "upper left")
    plt.title('Train_splits')
    plt.xlabel('Split')
    plt.ylabel('Distribution')
    plt.savefig("Train_splits.count - Cross validation.png")
    plt.show()

    x2 = list(np.arange(1, len(Prop_test)+1))
    # Plot results - test
    plt.bar(x2, Prop_test, 0.6, color='r', label = "elec")
    plt.bar(x2, E_mes_test, 0.6, bottom=Prop_test, color='b')
    plt.legend(loc = "upper left")
    plt.title('Test_splits')
    plt.xlabel('Split')
    plt.ylabel('Distribution')
    plt.savefig("Test_splits.count - Cross validation.png")
    plt.show()

    #Plot distribution of elec and window count in splits (scatter)
    try:
        C = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'lime', 'indigo', 'violet']
        for k in range(len(split_train_plot_F)) :
            plt.scatter(split_train_plot_F[k], split_train_elec_F[k], color=C[k])
            plt.scatter(split_test_plot_F[k], split_test_elec_F[k], color=C[k])
        plt.xlabel('Window_count')
        plt.ylabel('elec_count')
        plt.savefig("Splits_scatter.png")
        plt.show()
    except:
        print("The plot sucks - max splits is 10")

    #Plot distrubution of gender in outter layer of splits
    y1 = np.array(male_count_train_F)
    y2 = np.array(female_count_train_F)
    y3 = np.array(unknown_count_train_F)
    plt.bar(x, y1, color='r')
    plt.bar(x, y2, bottom=y1, color='b')
    plt.bar(x, y3, bottom=y1 + y2, color='y')
    plt.xlabel("Splits")
    plt.ylabel("Patients")
    plt.legend(["Male", "Female", "Unknown"])
    plt.title("Train_splits gender distribution - Cross validation")
    plt.savefig("Train_splits gender distribution - Cross validation.png")
    plt.show()

    y1 = np.array(male_count_test_F)
    y2 = np.array(female_count_test_F)
    y3 = np.array(unknown_count_test_F)
    plt.bar(x, y1, color='r')
    plt.bar(x, y2, bottom=y1, color='b')
    plt.bar(x, y3, bottom=y1 + y2, color='y')
    plt.xlabel("Splits")
    plt.ylabel("Patients")
    plt.legend(["Male", "Female", "Unknown"])
    plt.title("Test_splits gender distribution - Cross validation")
    plt.savefig("Test_splits gender distribution - Cross validation.png")
    plt.show()

    #Plot distribution of age in splitss
    bins = np.linspace(0, 100, 100)
    for i in range(len(x)):
        plt.hist(age_train_F[i], bins, label = i)
    plt.legend(loc='upper right')
    plt.xlabel("Age")
    plt.ylabel("Occurrence")
    plt.savefig("Train_splits age distribution - Cross validation.png")
    plt.show()

    for i in range(len(x)):
        plt.hist(age_test_F[i], bins, label=i)
    plt.legend(loc='upper right')
    plt.xlabel("Age")
    plt.ylabel("Occurrence")
    plt.savefig("Test_splits age distribution - Cross validation.png")
    plt.show()

    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), std(outer_results)))
    print('f1 score: %.3f (%.3f)' % (np.mean(outer_results_f1), std(outer_results_f1)))
    print('Balanced accuracy: %.3f (%.3f)' % (np.mean(outer_results_BA), std(outer_results_BA)))
    # report the best configuration
    print('Best Config based in acc: %s for model %s' % (best_model_params, best_model_))

    return [np.mean(outer_results), std(outer_results), best_model_]
    
if __name__ == "__main__":
    pass
    #Xtrain, Xtest, ytrain, ytest = splitDataset(TUH.index_patient_df, ratio=0.2, shuffle=True)