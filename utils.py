"""
helper functions
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

LOGGER = logging.getLogger(os.path.basename(__file__))

def show_example(image):
    """reshapes image to 100x100, plots)"""
    plt.imshow(image.reshape(100,100))
    plt.show()


def get_y_map(data):
    """gets all the unique values in y to allow str <-> int conversion"""
    y_map = LabelEncoder()
    y_map.fit(data['y']['train'])
    return(y_map)


def convert_y(y, y_map):
    """converts all y in data to int if str, else str if int"""
    # convert integers to string labels
    if np.issubdtype(type(y[0]), np.number):
        return(y_map.inverse_transform(y))

    # convert string labels to integers
    else:
        return(y_map.transform(y))


def write_results(output, y):
    """writes the vector y (in string form) to a file that kaggle accepts"""

    with open(output, 'w') as f:
        for i, val in enumerate(y):
            f.write('{},{}\n'.format(i, val))


def load_data(test_mode=False, valid_pct=0.1):
    """loads the data into a structure"""
    X_train = np.load('data/train_images.npy', encoding='latin1')
    X_test  = np.load('data/test_images.npy', encoding='latin1')
    y_train = np.genfromtxt('data/train_labels.csv', names=True, delimiter=',',
        dtype=[('Id', 'i8'), ('Category', 'S5')])

    # get data into numpy matrices
    X_train_output, y_train_output, X_test_output = [], [], []

    if test_mode:
        LOGGER.info('running in test mode, n=500')
        n_samples = 500
    else:
        n_samples = len(X_train)

    for i in range(n_samples):
       X_train_output.append(X_train[i, 1])
    X_train = np.vstack(X_train_output)

    for i in range(n_samples):
        y_train_output.append(np.array(y_train[i][1]).astype(np.str))
    y_train = np.hstack(y_train_output)

    for i in range(len(X_test)):
       X_test_output.append(X_test[i, 1])
    X_test = np.vstack(X_test_output)

    # make validation set
    n_valid = int(np.floor(valid_pct * n_samples))

    X_valid = X_train[:n_valid, :]
    X_train = X_train[n_valid:, :]
    y_valid = y_train[:n_valid]
    y_train = y_train[n_valid:]

    # data is accessed as data['X']['valid']
    data = {'X': {'train': X_train, 'valid': X_valid, 'test': X_test},
            'y': {'train': y_train, 'valid': y_valid}
    }

    LOGGER.debug('n TRAIN = {}, n VALID = {}, n TEST = {}'.format(
        X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

    return(data)

