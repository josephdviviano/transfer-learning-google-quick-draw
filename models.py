"""
holds our models (e.g., imagenet, cnns, etc, to be imported into experiments.py)
"""
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import logging
import os
import torch.nn as nn
import torch.optim as optim
from imutils import paths
import numpy as np
import argparse
import imutils
import os
import cv2

LOGGER = logging.getLogger(os.path.basename(__file__))

# global settings for all cross-validation runs
SETTINGS = {
    'n_cv': 25,
    'n_inner': 3,
}

# controls how chatty RandomizedCV is
VERB_LEVEL = 0


def set_parameter_requires_grad(model, fine_tune):
    """turns off gradient updates if we aren't fine tuning"""
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False


def inception_v3():

    model = models.inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, 31) # 31 classes for this problem

    # convert to PIL image before transforms for compatibility
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    return(model, transform)


def resnet(fine_tune=True, lr=10e-4, momentum=0.9, l2=0.01):
    """initializes a resnet that can be fine-tuned"""

    LOGGER.info('initializing resnet with fine_tuning={}'.format(fine_tune))

    model = models.resnet101(pretrained=True)     # resnet18,34,50,101,152

    # upper layers are set to requires_grad if fine_tune is true
    set_parameter_requires_grad(model, fine_tune)

    # this new layer is always trained
    linear_in_features = model.fc.in_features
    model.fc = nn.Linear(linear_in_features, 31) # 31 output classes
    torch.nn.init.xavier_uniform_(list(model.fc.parameters())[0]) # init weights

    # convert to PIL image before transforms for compatibility
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # if fine_dune is set to true, sets all parameters require_grad
    # otherwise only the newly added layers (above) require_grad
    params_to_update = model.parameters()

    LOGGER.debug("Params to learn:")
    if fine_tune:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                LOGGER.debug("\t{}".format(str(name)))
    else:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                LOGGER.debug("\t{}".format(str(name)))

    # observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=l2)

    return(model, transform, optimizer)

def k_nn():
    LOGGER.debug('building K-NN model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(10, 1000),
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1),
        'clf__n_neighbors': stats.randint(1,50)
    }
    
    # model we will train in our pipeline
    clf = KNeighborsClassifier()

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )
    
    return model



def SVM_nonlinear(data):
    """soft SVM with kernel"""
    LOGGER.debug('building SVM model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(10, 1000),
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1)
    }

    # model we will train in our pipeline
    clf = SVC(gamma=0.001, max_iter=100)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return(model)



def SVM(data):
    """ baseline: linear classifier (without kernel)"""
    LOGGER.debug('building SVM model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(10,1000),
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1)
    }

    # model we will train in our pipeline
    clf = SVC(kernel='linear', max_iter=100)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return(model)


def logistic_regression(data):
    """baseline: linear classifier"""
    LOGGER.debug('building logistic regression model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(10, 400),
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 10),
        'clf__penalty': ['l1', 'l2']
    }

    # model we will train in our pipeline
    clf = LogisticRegression(solver='saga', multi_class='ovr', max_iter=100)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return(model)


