"""
holds different experiment functions (import and run these in train.py)
"""
import matplotlib.pyplot as plt
from copy import copy
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
import models
import os
import time
import torch

LOGGER = logging.getLogger(os.path.basename(__file__))
SETTINGS = {'folds': 5, 'batch_size': 32, 'epochs': 100}
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

    loader_opts = {'batch_size': SETTINGS['batch_size'], 'num_workers': 0}

    # don't shuffle test set or our predictions will be wrong...
    train = DataLoader(train, shuffle=True,  **loader_opts)
    valid = DataLoader(valid, shuffle=True,  **loader_opts)
    test  = DataLoader(test,  shuffle=False, **loader_opts)

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


def make_X_proc(X, transform):
    """
    makes X (a minibatch) correct dimension for convnet (monochrome -> rgb)
    """
    X_proc = torch.zeros([X.shape[0], 3, 224, 224])

    for i in range(X.shape[0]):
        img = transform(X[i, :, :].unsqueeze(0))
        X_proc[i, 0, :, :] = img
        X_proc[i, 1, :, :] = img
        X_proc[i, 2, :, :] = img

    return(X_proc)


def pytorch_train_loop(model, transform, optimizer, train, valid):
    """
    Trains a submitted model with a given optimizer and scheduler.
    """
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=10)

    n_train = len(train.dataset)
    n_valid = len(valid.dataset)

    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []

    if CUDA:
        model = model.cuda()

    best_valid_loss = 10000 # big enough I think
    valid_loss = 10000      #

    # epochs
    for ep in range(SETTINGS['epochs']):

        # TRAIN
        scheduler.step(valid_loss)
        model.train(True)  # Set model to training mode
        total_loss, total_correct = 0.0, 0.0

        # minibatches
        for batch_idx, (X_train, y_train) in enumerate(train):

            optimizer.zero_grad()

            X_proc = make_X_proc(X_train, transform)

            if CUDA:
                X_proc, y_train = X_proc.cuda(), y_train.cuda()

            X_proc, y_train = Variable(X_proc), Variable(y_train)

            # do a forward-backward pass
            outputs = model.forward(X_proc)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # TRAIN: keep score
            n_correct = torch.sum(preds == y_train.data)
            total_loss += loss.item()
            total_correct += n_correct

        # error analysis
        LOGGER.debug('last outputs:\n{}'.format(outputs))
        LOGGER.debug('last predictions:\n{}'.format(preds))
        LOGGER.debug('last reality:\n{}'.format(y_train.data))

        # training performance
        train_loss = total_loss / (batch_idx+1)
        train_acc = total_correct.cpu().data.numpy() / n_train
        LOGGER.debug('epoch correct: {}/{}'.format(total_correct, n_train))
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())

        # VALID
        model.eval()
        total_loss, total_correct = 0.0, 0.0

        for batch_idx, (X_valid, y_valid) in enumerate(valid):

            # data management
            X_proc = make_X_proc(X_valid, transform)

            if CUDA:
                X_proc, y_valid = X_proc.cuda(), y_valid.cuda()

            X_proc, y_valid = Variable(X_proc), Variable(y_valid)

            # make predictions
            outputs = model.forward(X_proc)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, y_valid)

            # VALID: keep score
            n_correct = torch.sum(preds == y_valid.data)
            total_loss += loss.item()
            total_correct += n_correct

        # validation performance
        valid_loss = total_loss / (batch_idx+1)
        valid_acc = total_correct.cpu().data.numpy() / n_valid

        if valid_loss < best_valid_loss:
            best_epoch = ep
            best_valid_loss = valid_loss
            LOGGER.info('new best model found: loss={}'.format(valid_loss))
            best_model = model.state_dict()

        LOGGER.debug('VALID epoch correct: {}/{}'.format(total_correct, n_valid))
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc.item())

        LOGGER.info('[{}/100] loss={:.4f}/{:.4f}, acc={:.4f}/{:.4f}'.format(
                ep+1, train_loss, valid_loss, train_acc, valid_acc))


    # early stopping: use best validation performance
    LOGGER.info('early stopping -- rewinding to epoch {}'.format(best_epoch))
    model.load_state_dict(best_model)

    performance = {'train': {'accuracy': train_accs, 'loss': train_losses},
                   'valid': {'accuracy': valid_accs, 'loss': valid_losses},
                   'best_epoch': best_epoch
    }

    return(model, performance)


def resnet(data):
    """inception net"""

    train, valid, test = make_torch_loaders(data)

    # grid search
    lrs = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5]
    momentums = [0.6, 0.9, 0.95]
    l2s = [10e-2, 10e-3, 10e-5]
    best_val = 1000 # i think this is large enough

    for lr in lrs:
        for momentum in momentums:
            for l2 in lrs:

                model, transform, optimizer = models.resnet(
                    fine_tune=True, lr=lr, momentum=momentum, l2=l2)

                this_model, this_performance = pytorch_train_loop(
                    model, transform, optimizer, train, valid)

                # keep the best model
                best_epoch = this_performance['best_epoch']
                this_loss = this_performance['valid']['loss'][best_epoch]

                LOGGER.info('lr={}, momentum={}, l2={}, loss={:.4f}'.format(
                    lr, momentum, l2, this_loss))

                if this_loss < best_val:
                    best_model = copy(this_model)
                    best_performance = copy(this_performance)
                    best_optimizer = copy(optimizer)

    try:
        LOGGER.info('best optimizer: {}'.format(best_optimizer.state_dict()['param_groups']))
    except:
        pass

    # keep the best model across the grid search
    model = copy(best_model)
    performance = copy(best_performance)
    optimizer = copy(best_optimizer)

    # TEST: make predictions
    model.eval()
    test_predictions = []
    for batch_idx, X_test in enumerate(test):

        # data management
        X_proc = make_X_proc(X_test[0], transform)

        if CUDA:
            X_proc = X_proc.cuda()

        X_proc = Variable(X_proc)
        outputs = model.forward(X_proc)
        _, preds = torch.max(outputs.data, 1)
        test_predictions.extend(preds.tolist())

    test_predictions = np.array(test_predictions)

    #plt.plot(np.random.uniform(np.arange(100)))
    #plt.savefig('figures/test.jpg')
    #plt.close()

    #plt.plot(train_loss)
    #plt.plot(valid_loss)
    #plt.savefig('figures/resnet_loss.jpg')
    #plt.close()

    #plt.plot(train_acc)
    #plt.plot(valid_acc)
    #plt.savefig('figures/resnet_acc.jpg')
    #plt.close()

    return(test_predictions)


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


