# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from matplotlib.colors import ListedColormap



def report_classification_result(classifier_name, ytrue, pred):

    print("Confusion Matrix of ", classifier_name, ": \n")
    cm = pd.DataFrame(confusion_matrix(ytrue, pred))
    display(cm)
    print("\n")

    print("Precision, Recall, and F-1 score of ", classifier_name, ": \n")
    print(classification_report(ytrue, pred, digits=4))




def show_decision_boundary(clf, X_r, Y, h_value):

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])


    # create coordinate for the contour plot
    h = .02
    x_min, x_max = X_r[:, 0].min() - .5, X_r[:, 0].max() + .5
    y_min, y_max = X_r[:, 1].min() - .5, X_r[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf.fit(X_r, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    plt.figure(figsize=(4, 4))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.4)

    # display data points with original labels
    idx = np.where(Y == 0)[0]
    plt.scatter(X_r[idx, 0], X_r[idx, 1], c='#FF0000', label=0, s=6)

    idx = np.where(Y == 1)[0]
    plt.scatter(X_r[idx, 0], X_r[idx, 1], c='#0000FF', label=1, s=6)

    plt.legend(loc="best")
    plt.title("Decision Boundary,  h =" + str(h_value))
    plt.show()