# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:52:20 2018

@author: leaar
"""

#!/usr/bin/env python

import os, sys
import numpy as np
import torch
import sklearn
import pandas as pd
import seaborn as sns
from numpy import genfromtxt


def make_confusion_mat(mat, title, output, label):
    plot = sns.heatmap(mat, annot=False, linewidths=0.5, cmap='Blues')
    plot.set_xticklabels(label, rotation=45)
    plot.set_yticklabels(label, rotation=45)
    plot.set_xlabel('Actual class')
    plot.set_ylabel('Predicted class')
    plot.set_title(title)
    fig = plot.get_figure()
    fig.savefig(output)
    fig.clear()
    

my_data = genfromtxt('confusion_lr.csv', delimiter=',')
label = ['apple', 'empty', 'moust', 'mouth', 'mug', 'nail', 'nose', 'octag', 'paint', 'panda', 'parro', 'peanu', 'pear', 'pencil', 'pengu', 'pillo', 'pinea', 'pool', 'rabbi', 'rhino', 'rifle', 'rolle', 'sailb', 'scorp', 'screw', 'shove', 'sink', 'skate', 'skull', 'spoon']

make_confusion_mat(my_data, 'confusion matrix logistic regression model', 'confusion_lr_2.png' ,label)