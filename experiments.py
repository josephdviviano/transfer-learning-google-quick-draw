"""
holds different experiment functions (import and run these in train.py)
"""

from copy import copy
from models import SVM
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import logging
import os

LOGGER = logging.getLogger(os.path.basename(__file__))
SETTINGS = {'folds': 10}

def experiment01():
    """imagenet"""
    pass


def experiment02():
    """imagenet + PrELU"""
    pass


def experiment03():
    """adaboost logistic regression"""
    pass


def experiment04():
    """cnn and random forest ensemble"""
    pass


def svm_baseline(data):
    """baseline: SVM (without Kernel)"""

    X_train = data['X']['train']
    y_train = data['y']['train']
    X_valid = data['X']['valid']
    y_valid = data['y']['valid']
    X_test  = data['X']['test']

    model = SVM(data) # returns a model ready to train

    kf = StratifiedKFold(n_splits=SETTINGS['folds'], shuffle=True)

    best_model_acc = -1
    for i, (train_idx, test_idx) in enumerate(kf.split(X_train, y_train)):
        LOGGER.info("fold {}/{}".format(i+1, SETTINGS['folds']))

        # split training and test sets
        X_fold_train = X_train[train_idx]
        X_fold_test  = X_train[test_idx]
        y_fold_train = y_train[train_idx]
        y_fold_test  = y_train[test_idx]

        # fit model
        model.fit(X_fold_train, y_fold_train)

        this_model_predictions = model.predict(X_fold_test)
        this_model_acc = accuracy_score(this_model_predictions, y_fold_test)

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


