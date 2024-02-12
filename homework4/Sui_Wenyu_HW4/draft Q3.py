import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from IPython.display import display

# read data
divorce = pd.read_csv('data\marriage.csv', header = None)
X = np.array(divorce)[:, 0:-1]
Y = np.array(divorce)[:, -1]

# split data into train and test
# use the first 80% data for training and the remaining 20% for testing
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2,
                                                #random_state = 2
                                                shuffle = False
                                                )

names = ["Naive Bayes", "Logistic Regression", "KNN"]

classifiers = [GaussianNB(),
               LogisticRegression(max_iter=200, solver = 'liblinear').fit(xtrain, ytrain),
               KNeighborsClassifier(3)]

# compare the performance of different classifiers
for name, clf in zip(names, classifiers):

    # fit the model
    model = clf.fit(xtrain, ytrain)

    # calculate testing accuracy
    # ## test error
    ypred_test = clf.predict(xtest)
    matched_test = ypred_test == ytest
    acc_test = sum(matched_test) / len(matched_test)

    print("Testing Accuracy of ", name, ":", round(acc_test,4))





# perform PCA to the origianl dataset
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# display data points with original labels
plt.figure()
plt.title("Two-dimensional PCA Results with Original Labels")

idx = np.where(Y == 0)[0]
plt.scatter(X_r[idx, 0], X_r[idx, 1], c='#FF0000', label = 0)

idx = np.where(Y == 1)[0]
plt.scatter(X_r[idx, 0], X_r[idx, 1], c='#0000FF', label = 1)

plt.legend(loc = "best")
plt.show()

# create coordinate for the contour plot
h = .02
x_min, x_max = X_r[:, 0].min() - .5, X_r[:, 0].max() + .5
y_min, y_max = X_r[:, 1].min() - .5, X_r[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

for name, clf in zip(names, classifiers):
    clf.fit(X_r, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    plt.figure()
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.4)

    # display data points with original labels
    idx = np.where(Y == 0)[0]
    plt.scatter(X_r[idx, 0], X_r[idx, 1], c='#FF0000', label=0)

    idx = np.where(Y == 1)[0]
    plt.scatter(X_r[idx, 0], X_r[idx, 1], c='#0000FF', label=1)

    plt.legend(loc="best")
    plt.title("Decision Boundary of " + name)
    plt.show()