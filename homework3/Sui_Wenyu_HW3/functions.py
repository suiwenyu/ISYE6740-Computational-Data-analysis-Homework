# import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.neighbors import KernelDensity
from scipy.sparse import csc_matrix
from scipy import stats




def TwoD_histogram(data, nbin, xlabel, ylabel):
    # for 2 dimensional data

    plt.figure()
    plt.hist2d(data[:, 0], data[:, 1], bins=nbin)
    plt.title('2D Histogram, bins on each axis: ' + str(nbin))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def show_image(width, height, pixels):
    pixels = pixels.reshape(height, width)
    pixels = np.fliplr(pixels)
    pixels = np.rot90(pixels)
    plt.imshow(pixels, cmap = 'gray')
    plt.show()



def MNIST_true_labels_and_purity(Class, y):
    labels = []
    purities = []
    true_counts = []
    assigned = []
    mis_classification_rates = []

    for i in range(2):
        #  find all data points that belong to class i
        idx = np.where(Class == i)[0]
        ysubgroup = y[idx]
        true_counts += [ysubgroup.shape[0]]

        unique, counts = np.unique(ysubgroup, return_counts=True)

        label = unique[np.argmax(counts)]
        labels += [label]

        assigned += [counts[np.argmax(counts)]]
        purity = counts[np.argmax(counts)] / ysubgroup.shape[0]
        purities += [purity]
        mis_classification_rates += [1 - purity]

    return labels, assigned, true_counts, purities, mis_classification_rates




def k_means_MNIST_l2(pixels, k):
    m = pixels.shape[0]
    # run kmeans;
    # Number of clusters.
    cno = k
    x = pixels

    # Randomly initialize centroids with data points;
    c = x[np.random.randint(x.shape[0], size=(1, cno))[0], :]
    c_old = c.copy() + 10

    while np.linalg.norm(c - c_old, ord='fro') > 1e-8:
        c_old = c.copy()

        # norm squared of the centroids;
        c2 = np.sum(np.power(c, 2), axis=1, keepdims=True)

        # For each data point x, computer min_j  -2 * x' * c_j + c_j^2;
        # Note that here is implemented as max, so the difference is negated.
        tmpdiff = (2 * np.dot(x, c.T) - c2.T)
        labels = np.argmax(tmpdiff, axis=1)

        # Update data assignment matrix;
        # The assignment matrix is a sparse matrix,
        # with size m x cno. Only one 1 per row.
        P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, cno))
        count = P.sum(axis=0).T

        # Recompute centroids;
        c = np.array((P.T.dot(x)) / count)

    Class = labels
    centroid = c

    return Class, centroid