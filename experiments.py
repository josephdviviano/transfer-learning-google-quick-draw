"""
holds different experiment functions (import and run these in train.py)
"""

import os
import logging
from models import SVM
from scipy import stats
from sklearn.metrics import accuracy_score

LOGGER = logging.getLogger(os.path.basename(__file__))

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


def experiment05(data):
    """baseline: SVM (without Kernel)"""

    X_train = data['X']['train']
    y_train = data['y']['train']

    model = SVM(data) # returns a model ready to train

    model.fit(data['X']['train'], data['y']['train']) # fit the training data
    y_valid_pred = model.predict(data['X']['valid'])  # validation scores
    y_test_pred = model.predict(data['X']['test'])    # test scores

    LOGGER.info('valid accuracy: {}'.format(y_valid_pred, data['y']['valid']))
    import IPython; IPython.embed()

    return(y_test_pred, model)


