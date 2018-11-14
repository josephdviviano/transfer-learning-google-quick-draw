"""
holds our models (e.g., imagenet, cnns, etc, to be imported into experiments.py)
"""
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import logging
import os
import torch
import torch.nn.functional as func
import torchvision


LOGGER = logging.getLogger(os.path.basename(__file__))

# stores downloaded models
CACHEDIR = os.path.expanduser(os.path.join('~', '.torch'))

# global settings for all cross-validation runs
SETTINGS = {
    'n_cv': 100,
    'n_inner': 3,
}


def resnet50():
    if not exists(cache_dir):
        makedirs(cache_dir)

    models_dir = cache_dir + '/' + 'models/'
    if not exists(models_dir):
        makedirs(models_dir)

    model_name = 'resnet50-19c8e357.pth'
    src = '../input/pretrained-pytorch-models/' + model_name;
    dest = models_dir + model_name
    copyfile(src, dest)


def SVM(data):
    """ baseline: linear classifier (without kernel)"""
    # hyperparameters to search for randomized cross validation
    settings = {
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1)
    }

    # model we will train in our pipeline
    clf = SVC(kernel='linear', max_iter=100)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=2,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return(model)


def logistic_regression(data):
    """Base line: Linear classifier"""
    pass


