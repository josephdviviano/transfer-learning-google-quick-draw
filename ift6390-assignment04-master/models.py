"""
holds our models (e.g., imagenet, cnns, etc, to be imported into experiments.py)
"""
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import ensemble
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision.models as models
import logging
import os
import torch
import torch.nn.functional as func
import torchvision
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier




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


def resnet(fine_tune=True):
    """initializes a resnet that can be fine-tuned"""

    LOGGER.info('initializing resnet with fine_tuning={}'.format(fine_tune))

    # upper layers are trainied if fine_tune=True
    model = models.resnet18(pretrained=True) # resnet34,50,101,152
    set_parameter_requires_grad(model, fine_tune) # sets weight plasticity

    # this new layer is always trained
    model.fc = nn.Linear(512, 31) # 31 classes
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

    # send the model to GPU
    #model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    # finetuning we will be updating all parameters. However, if we are
    # doing feature extract method, we will only update the parameters
    # that we have just initialized, i.e. the parameters with requires_grad
    # is True.
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

    print(params_to_update)

    # observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=10e-4, momentum=0.9)

    return(model, transform, optimizer)


def SVM(data):
    """ baseline: linear classifier (without kernel)"""
    LOGGER.debug('building SVM model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(10, 1000),
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1)
    }

    # model we will train in our pipeline
    clf = SVC(kernel='rbf', max_iter=500)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return(model)


def logistic_regression(data):
    """baseline: linear classifier"""
    LOGGER.debug('building logistic regression model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(10, 1000),
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1),
        'clf__penalty': ['l1', 'l2']
    }

    # model we will train in our pipeline
    clf = LogisticRegression(solver='saga', multi_class='ovr', max_iter=200)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return(model)


def gbm(data):
    LOGGER.debug('building gradient boosting machine model')
    # hyperparameters to search for randomized cross validation
    settings = {
            'dim__n_components': stats.randint(10, 1000),
            "clf__loss":["deviance"],
            "clf__learning_rate":[0.01], #np.linspace(0.001, 0.15, 20),
            "clf__n_estimators":[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "clf__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            "clf__criterion": ["friedman_mse",  "mae"],
            "clf__min_samples_split": np.linspace(0.1, 1.0, 12),
            "clf__min_samples_leaf": [1,2,3,4,5,6,7,8,9,10],
            "clf__max_depth":[1,2,3,4,5,6,7,8,9,10],
            "clf__max_features":["log2","sqrt"]
    }

    # model we will train in our pipeline
    clf = ensemble.GradientBoostingClassifier()
    
    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf)
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )
    
    return(model)


def xgb(data):
    LOGGER.debug('building X gradient boosting machine model (GPU version)')
    # hyperparameters to search for randomized cross validation
    settings = {
            'dim__n_components': stats.randint(10, 1000),
            "clf__max_depth":[1,2,3,4,5,6,7,8,9,10]	,
            "clf__learning_rate": [.05, 0.1, 0.5],
            "clf__n_estimators":[1, 2, 4, 8, 10, 16, 50, 100],
            "clf__objective":'multi:softmax',
            'clf__n_jobs':[1],
            "clf__min_child_weight": np.linspace(0.1, 1.0, 12),
            "clf__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            "clf__silent": [True]            
    }

    # model we will train in our pipeline
    clf = XGBClassifier()
    
    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf)
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )
    
    return(model)



