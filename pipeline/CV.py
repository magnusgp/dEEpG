#Make two level CV for model selection
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

def CrossValidtion_multi(model, X, Y, n_splits_outer=10, n_splits_inner=5, random_state=None):
    cv_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    outer_results = list()
    for train_index, test_index in cv_outer.split(X):
        #Split the data
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
        # define search space
        space = dict()
        space['n_estimators'] = [10, 100, 500]
        space['max_features'] = [2, 4, 6]
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
    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
    return mean(outer_results), std(outer_results)

def Crossvalidation_simple(models, X, Y, n_splits=10, random_state=None):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = list()
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
            results.append(acc)
            # summarize the results
            print('>%s: %.3f' % (name, acc))
    # summarize the average accuracy
    for name, model in models.items():
        print('%s: %.3f' % (name, mean(results)))
    return name, mean(results)