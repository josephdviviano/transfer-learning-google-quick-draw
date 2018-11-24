plt.plot(performance['train']['loss'])
plt.plot(performance['valid']['loss'])
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss (Cross Entropy)')
plt.title('Training and Validation Loss')
plt.axvline(performance['best_epoch'])
plt.savefig('figures/training_loss.jpg')
plt.close()
plt.clf()

plt.plot(performance['train']['accuracy'])
plt.plot(performance['valid']['accuracy'])
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss (Cross Entropy)')
plt.title('Training and Validation Accuracy')
plt.axvline(performance['best_epoch'])
plt.savefig('figures/training_accuracy.jpg')
plt.close()
plt.clf()

import os, sys
import numpy as np
import torch
import sklearn
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(performance['best_preds'], performance['best_reals'])
label = ['apple', 'empty', 'moust', 'mouth', 'mug', 'nail', 'nose', 'octag', 'paint', 'panda', 'parro', 'peanu', 'pear', 'pencil', 'pengu', 'pillo', 'pinea', 'pool', 'rabbi', 'rhino', 'rifle', 'rolle', 'sailb', 'scorp', 'screw', 'shove', 'sink', 'skate', 'skull', 'spoon']
plot = sns.heatmap(mat, annot=False, linewidths=0.5, cmap='Blues', xticklabels=label, yticklabels=label)
#plot.set_xticks(np.arange(len(label)), minor=True)
#plot.set_yticks(np.arange(len(label)), minor=True)
#plot.set_xticklabels(label, rotation=45, minor=True)
#plot.set_yticklabels(label, rotation=45, minor=True)
plot.set_xlabel('Actual class')
plot.set_ylabel('Predicted class')
plot.set_title('Confusion Matrix for Resnet')
fig = plot.get_figure()
fig.savefig('figures/resnet_confusionmat.jpg')
fig.clear()
fig.clf()



