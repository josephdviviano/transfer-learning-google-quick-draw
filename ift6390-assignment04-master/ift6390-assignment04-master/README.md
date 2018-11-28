ift6390-assignment04
--------------------

+ **data/download.sh**: gets the data (requires the [kaggle api](https://github.com/Kaggle/kaggle-api))
+ **models.py**: a collection of pytorch, Scikit-learn and Xgboost models
+ **experiments.py** the application of some models from **models.py** to the data/, to produce predictions
+ **train.py**: runs a set of experiments from **experiments.py**, combines their predictions, and writes a kaggle submission
