import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix


# read data
matFile = sio.loadmat('./data/mnist_10digits')

xtrain = matFile['xtrain']
ytrain = matFile['ytrain'].flatten()
xtest = matFile['xtest']
ytest = matFile['ytest'].flatten()

# standardize data
xtrain = xtrain / 255
xtest = xtest / 255


# KNN
KNN = KNeighborsClassifier(n_neighbors = 7).fit(xtrain, ytrain)

pred_knn = KNN.predict(xtrain)
print(classification_report(ytest, pred_knn, digits = 4))