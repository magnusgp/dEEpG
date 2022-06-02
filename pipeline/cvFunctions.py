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

def CrossValidation_2(model, name, X, Y, n_splits_outer=10, n_splits_inner=5, random_state=None):
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

def CrossValidation_1(models, X, Y, n_splits=10, random_state=None):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results_acc = list()
    results_f1 = list()
    dict = {}
    dict_f1 = {}
    best_model = [[0, 0, 0]]
    for name, model in models.items():
        dict[name] = list()
        dict_f1[name] = list()
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