"""
holds different experiment functions (import and run these in train.py)
"""

from copy import copy
from models import SVM, logistic_regression
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import logging
import os
import time

LOGGER = logging.getLogger(os.path.basename(__file__))
SETTINGS = {'folds': 5}


def kfold_train_loop(data, model):
    """
    trains a model using stratified kfold cross validation. hyperparameter
    selection is expected to be performed inside the submitted model as part
    of the pipeline.
    """
    x_train = data['X']['train']
    y_train = data['y']['train']

    kf = StratifiedKFold(n_splits=SETTINGS['folds'], shuffle=True)

    best_model_acc = -1
    last_time = time.time()

    for i, (train_idx, test_idx) in enumerate(kf.split(x_train, y_train)):

        this_time = time.time()
        LOGGER.info("fold {}/{}, {:.2f} sec elapsed".format(
            i+1, SETTINGS['folds'], this_time - last_time))
        last_time = this_time

        # split training and test sets
        X_fold_train = X_train[train_idx]
        X_fold_test  = X_train[test_idx]
        y_fold_train = y_train[train_idx]
        y_fold_test  = y_train[test_idx]

        # fit model on fold (does all hyperparameter selection ox X_fold_train)
        model.fit(X_fold_train, y_fold_train)

        this_model_predictions = model.predict(X_fold_test)
        this_model_acc = accuracy_score(this_model_predictions, X_fold_test)

        if this_model_acc > best_model_acc:
            best_model = copy(model)

    best_model.fit(data['X']['train'], data['y']['train']) # fit training data
    y_train_pred = best_model.predict(data['X']['train'])  # train scores
    y_valid_pred = best_model.predict(data['X']['valid'])  # validation scores
    y_test_pred = best_model.predict(data['X']['test'])    # test scores

    LOGGER.info('train/valid accuracy: {}/{}'.format(
        accuracy_score(y_train_pred, data['y']['train']),
        accuracy_score(y_valid_pred, data['y']['valid'])
    ))

    results = {
        'train': y_train_pred, 'valid': y_valid_pred, 'test': y_test_pred
    }

    return(results, best_model)


def experiment01():
    """imagenet"""
    pass


def experiment02():
    """imagenet + PrELU"""
    pass


def experiment03():
    """adaboost logistic regression"""
    pass


def lr_baseline(data):
    """baseline: logistic regression"""

    model = logistic_regression(data) # returns a model ready to train
    results, best_model = kfold_train_loop(data, model)

    return(results, best_model)


def svm_baseline(data):
    """baseline: SVM (without Kernel)"""

    model = SVM(data) # returns a model ready to train
    results, best_model = kfold_train_loop(data, model)

    return(results, best_model)


