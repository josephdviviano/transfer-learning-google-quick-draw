"""
holds our models (e.g., imagenet, cnns, etc, to be imported into experiments.py)
"""

from sklearn.svm import SVC


def SVM(data):
    """ Base line: Linear classifier (without kernel)"""
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(data['X']['train'], data['y']['train'])

    y_pred = svclassifier.predict(data['X']['valid'])

    return(y_pred)


def logistic_regression(data):
    """Base line: Linear classifier"""
    pass


