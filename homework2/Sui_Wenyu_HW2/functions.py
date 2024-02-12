import pandas as pd
import numpy as np
import scipy.sparse.linalg as ll
import math
from IPython.display import display
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import shortest_path
from PIL import Image
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



def pca_food_consumption(x):
    # perform PCA
    # standardize each feature variable
    m = x.shape[0]
    stdx = np.std(x, axis=0)
    x = x @ np.diag(np.ones(stdx.shape[0]) / stdx)
    x = x.T
    mu = np.mean(x, axis=1)
    xc = x - mu[:, None]

    # covariance matrix
    C = np.dot(xc, xc.T) / m

    # find 2 eigenvectors with largest eigenvalues
    K = 2
    S, W = np.linalg.eig(C)
    S = S.real
    W = W.real

    # sort eigenvalues
    idx = np.argsort(-S)
    S = S[idx][0: K]
    W = W[:, idx][:, 0: K]

    dim1 = np.dot(W[:, 0].T, xc) / math.sqrt(S[0])  # extract 1st eigenvalues
    dim2 = np.dot(W[:, 1].T, xc) / math.sqrt(S[1])  # extract 2nd eigenvalue

    return dim1, dim2

def visual_food_consumption(dim1, dim2, labels):
    color_string = 'bgrmck'
    marker_string = '.+*o'
    fig = plt.figure()
    for i in range(int(labels.shape[0])):
        color = color_string[i % 5]
        marker = marker_string[i % 4]
        m = color + marker
        plt.plot(dim1[i], dim2[i], m)
        plt.text(dim1[i], dim2[i], labels[i])
    plt.show()



def isomap_adjacency_matrix_epsilon_threshold(A, epsilon):
    A1 = A.copy()
    x = np.where(A1 > epsilon)[0]
    y = np.where(A1 > epsilon)[1]

    for i, j in zip(x, y):
        A1[i, j] = 0

    return A1



def isomap(A, eps):
    A1 = isomap_adjacency_matrix_epsilon_threshold(A, eps)
    #plt.imshow(A1, plt.cm.binary)
    #plt.show()

    # Implement ISOMAP
    # find the shortest between data points
    m = A1.shape[0]
    D = shortest_path(A1, method = "D")
    D[np.isinf(D)] = 99999

    H = np.identity(m) - np.ones((m,m))/m
    C = -0.5 * H.dot(np.power(D, 2)).dot(H)

    # find 2 eigenvectors with largest eigenvalues
    K = 2
    S, W = np.linalg.eig(C)

    # sort eigenvalue
    idx = np.argsort(-S)
    S = S[idx][0:K]
    W = -1 * W[:, idx][:,0:K]

    Z = W.dot(np.diag(np.sqrt(S)))

    return Z



def pca_isomap(x):
    # perform PCA
    m = x.shape[0]

    x = x.T
    mu = np.mean(x, axis=1)
    xc = x - mu[:, None]

    # covariance matrix
    C = np.dot(xc, xc.T) / m

    # find 2 eigenvectors with largest eigenvalues
    K = 2
    S, W = ll.eigs(C, k=K)
    S = S.real
    W = W.real

    # sort eigenvalues
    idx = np.argsort(-S)
    S = S[idx]
    W = W[:, idx]

    dim1 = np.dot(W[:, 0].T, xc) / math.sqrt(S[0])  # extract 1st eigenvalues
    dim2 = np.dot(W[:, 1].T, xc) / math.sqrt(S[1])  # extract 2nd eigenvalue

    return dim1, dim2




def read_and_resize_image_pixels(filepath):

    picture = Image.open(filepath, 'r')

    width, height = picture.size

    height = height // 4
    width = width // 4
    picture_resized = picture.resize((width, height))

    pixel_values = np.array(list(picture_resized.getdata()))

    return width, height, pixel_values



def show_image(width, height, pixels):
    pixels = pixels.reshape(height, width)
    plt.imshow(pixels, cmap = 'gray')
    plt.show()



def create_subject_face_matrix(subject_files):
    for file in subject_files:

        # read and downsize image
        width, height, pixel_values = read_and_resize_image_pixels("Data/yalefaces/" + file)

        # add image to matrix of faces
        pixel_values = pixel_values.reshape((1, pixel_values.shape[0]))
        try:
            faces = np.concatenate((faces, pixel_values), axis=0)
        except:
            faces = pixel_values

    return faces, width, height


def pca_eigenface(x):
    # perform PCA
    m = x.shape[0]

    x = x.T
    mu = np.mean(x, axis=1)
    xc = x - mu[:, None]

    # covariance matrix
    C = np.dot(xc, xc.T) / m

    # find 6 eigenvectors with largest eigenvalues
    K = 6
    S, W = np.linalg.eig(C)
    S = S.real
    W = W.real

    # sort eigenvalues
    idx = np.argsort(-S)
    S = S[idx][0: K]
    W = W[:, idx][:, 0: K]

    S_sqrt = np.identity(K) / np.diag(np.sqrt(S))
    S_sqrt = np.nan_to_num(S_sqrt, nan = 0)
    Z = S_sqrt.dot(W.T.dot(xc))

    return Z, W