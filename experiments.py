"""
holds different experiment functions (import and run these in train.py)
"""

from copy import copy
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import logging
import models
import os
import time
import torch

LOGGER = logging.getLogger(os.path.basename(__file__))
SETTINGS = {'folds': 5, 'batch_size': 50}
CUDA = torch.cuda.is_available()

def make_torch_loaders(data):
    """numpy --> tensor --> dataloader"""
    train = TensorDataset(
        torch.FloatTensor(data['X']['train']),
        torch.LongTensor(data['y']['train']))
    valid = TensorDataset(
        torch.FloatTensor(data['X']['valid']),
        torch.LongTensor(data['y']['valid']))
    test  = TensorDataset(
        torch.FloatTensor(data['X']['test']))

    loader_opts = {'batch_size': SETTINGS['batch_size'],
        'shuffle': True, 'num_workers': 2}

    train = DataLoader(train, **loader_opts)
    valid = DataLoader(valid, **loader_opts)
    test  = DataLoader(test,  **loader_opts)

    return(train, valid, test)


def kfold_train_loop(data, model):
    """
    trains a model using stratified kfold cross validation. hyperparameter
    selection is expected to be performed inside the submitted model as part
    of the pipeline.
    """
    X_train = data['X']['train']
    y_train = data['y']['train']

    kf = StratifiedKFold(n_splits=SETTINGS['folds'], shuffle=True)

    best_model_acc = -1
    last_time = time.time()

    for i, (train_idx, test_idx) in enumerate(kf.split(X_train, y_train)):

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


def inception_net(data):
    """inception net"""
    model, transform = models.inception_v3()
    import IPython; IPython.embed()


def resnet(data):
    """inception net"""
    model, transform, optimizer = models.resnet()
    n_train = data['X']['train'].shape[0]
    train, valid, test = make_torch_loaders(data)
    criterion = torch.nn.CrossEntropyLoss()

    if CUDA:
        model = model.cuda()

    # epochs
    for ep in range(10):

        #scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # minibatches
        for batch_idx, (X_train, y_train) in enumerate(train):

            optimizer.zero_grad()

            # makes inputs correct dimension for convnet
            X_train_proc = torch.zeros([SETTINGS['batch_size'], 3, 224, 224])
            for i in range(X_train.shape[0]):
                X_train_proc[i, :, :, :] = transform(X_train[i, :, :].view(-1, 1, 1))

            if CUDA:
                X_train_proc, y_train = X_train_proc.cuda(), y_train.cuda()

            X_train_proc, y_train = Variable(X_train_proc), Variable(y_train)

            # do a pass
            outputs = model.forward(X_train_proc)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # keep the score
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == y_train.data)

        #
        epoch_loss = running_loss / (batch_idx+1)
        epoch_acc = running_corrects / n_train
        logger.info('[{}/10] Loss: {:.4f} Acc: {:.4f}'.format(
                ep+1, epoch_loss, epoch_acc))

def svm_nonlinear(data):
    """baseline: SVM (without Kernel)"""

    model = models.SVM_nonlinear(data) # returns a model ready to train
    results, best_model = kfold_train_loop(data, model)

    return(results, best_model)

def lr_baseline(data):
    """baseline: logistic regression"""

    model = models.logistic_regression(data) # returns a model ready to train
    results, best_model = kfold_train_loop(data, model)

    return(results, best_model)


def svm_baseline(data):
    """baseline: SVM (without Kernel)"""

    model = models.SVM(data) # returns a model ready to train
    results, best_model = kfold_train_loop(data, model)

    return(results, best_model)


