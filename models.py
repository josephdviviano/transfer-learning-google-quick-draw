"""
holds our models (e.g., imagenet, cnns, etc, to be imported into experiments.py)
"""
def SVM(data):
    """ Base line: Linear classifier (without kernel)"""
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(data['X']['train'], data['y']['train']) 
    
    y_pred = svclassifier.predict(data['X']['valid'])  
    
    return y_pred
    

def logistic_regression(data):
    """Base line: Linear classifier"""
