from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from datetime import datetime
import math
from os.path import abspath, exists


# The following functions partially used the demo code from lecture 2 and lecture 3


def read_image_pixels(filepath):

    picture = Image.open(filepath, 'r')
    width, height = picture.size

    pixel_values = list(picture.getdata())
    pixel_values = np.array(pixel_values).reshape((width * height, 3))

    return width, height, pixel_values


def show_image(width, height, pixels):
    pixels = pixels.reshape(height, width, 3)
    plt.imshow(pixels)
    plt.show()


def k_means_compress(width, height, pixels, k):
    start_time = datetime.now()
    iterno = 0

    m = pixels.shape[0]
    # run kmeans;
    # Number of clusters.
    cno = k
    x = pixels

    # Randomly initialize centroids with data points;
    c = x[np.random.randint(x.shape[0], size=(1, cno))[0], :]
    c_old = c.copy() + 10

    while np.linalg.norm(c - c_old, ord='fro') > 1e-6:
        c_old = c.copy()
        iterno += 1

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

    end_time = datetime.now()
    time_to_converge = (end_time - start_time).total_seconds()

    Class = labels
    centroid = c

    # Calculate the sum squared Euclidian distance between each data point and the centroid of its cluster
    x_clustered = np.zeros(shape=(height * width, 3), dtype=int)
    for i in range(cno):
        x_clustered[np.where(Class == i)[0], :] = centroid[i, :]
    obj = np.linalg.norm(x - x_clustered, ord='fro')

    return Class, centroid, time_to_converge, iterno, obj




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

    # Calculate the sum squared Euclidian distance between each data point and the centroid of its cluster
    x_clustered = np.zeros(shape=(m, pixels.shape[1]))
    for i in range(cno):
        x_clustered[np.where(Class == i)[0], :] = centroid[i, :]
    obj = np.linalg.norm(x - x_clustered, ord='fro')

    return Class, centroid, obj




def k_means_MNIST_l1(pixels, k):
    m = pixels.shape[0]
    # run kmeans;
    # Number of clusters.
    cno = k
    x = pixels

    # Randomly initialize centroids with data points;
    c = x[np.random.randint(x.shape[0], size=(1, cno))[0], :]
    c_old = c.copy() + 10

    while np.sum(np.abs(c - c_old)) > 1e-6:
        c_old = c.copy()

        # For each data point x, computer nearest cntroid in l1 distance
        # 'labels' stores the assigned cluster
        labels = np.empty(shape=(m))

        # search for the nearest centroid
        for i in range(m):
            datapoint = x[i, :]
            tmpdiff = np.sum(np.abs(c - datapoint), axis=1, keepdims=True)

            label = np.argmin(tmpdiff, axis=0)
            labels[i] = label

        # Recompute centroids;
        for i in range(cno):
            c[i, :] = np.median(x[np.where(labels == i)[0], :], axis=0)

    Class = labels
    centroid = c

    # Calculate the sum squared Euclidian distance between each data point and the centroid of its cluster
    x_clustered = np.zeros(shape=(m, pixels.shape[1]))
    for i in range(cno):
        x_clustered[np.where(Class == i)[0], :] = centroid[i, :]
    obj = np.sum(np.power(np.abs(x - x_clustered),2))

    return Class, centroid, obj





def MNIST_true_labels_and_purity(Class, y):
    labels = []
    purities = []
    true_counts = []
    assigned = []

    for i in range(10):
        #  find all data points that belong to class i
        idx = np.where(Class == i)[0]
        ysubgroup = np.squeeze(y[:, idx])
        true_counts += [ysubgroup.shape[0]]

        unique, counts = np.unique(ysubgroup, return_counts=True)

        label = unique[np.argmax(counts)]
        labels += [label]

        assigned += [counts[np.argmax(counts)]]
        purity = counts[np.argmax(counts)] / ysubgroup.shape[0]
        purities += [purity]

    return labels, assigned, true_counts, purities




def display_compressed_image(width, height, k, Class, centroid):
    cno = k

    x_clustered = np.zeros(shape=(height * width, 3), dtype=int)
    for i in range(cno):
        x_clustered[np.where(Class == i)[0], :] = np.round(centroid[i, :])

    show_image(width, height, x_clustered)
    plt.show()


def show_NIST_image_function(centroids, H, W):
    N = int((centroids.shape[1]) / (H * W))
    assert (N == 3 or N == 1)

    # Organize the images into rows x cols.
    K = centroids.shape[0]
    COLS = round(math.sqrt(K))
    ROWS = math.ceil(K / COLS)

    COUNT = COLS * ROWS

    plt.clf()
    # Set up background at value 100 [pixel values 0-255].
    image = np.ones((ROWS * (H + 1), COLS * (W + 1), N)) * 100
    for i in range(0, centroids.shape[0]):
        r = math.floor(i / COLS)
        c = np.mod(i, COLS)

        image[(r * (H + 1) + 1):((r + 1) * (H + 1)), \
        (c * (W + 1) + 1):((c + 1) * (W + 1)), :] = \
            centroids[i, :].reshape((H, W, N))

    plt.imshow(image.squeeze(), plt.cm.gray)
    plt.show()




def political_read_edges(f_path):
    # read the graph
    f_path = abspath(f_path)
    if exists(f_path):
        with open(f_path) as graph_file:
            lines = [line.split() for line in graph_file]
    return np.array(lines).astype(int)



def political_read_blog_info(f_path):
    # read nodes file
    indices = []
    urls = []
    orientations = []
    if exists(f_path):
        with open(f_path) as fid:
            for line in fid.readlines():
                index = line.split("\t", 4)[0]
                url = line.split("\t", 4)[1]
                orientation = line.split("\t", 4)[2]

                indices += [index]
                urls += [url]
                orientations += [orientation]
    return indices, urls, orientations


def political_true_labels_and_purity(Class, y, k):
    labels = []
    mismatch_rates = []
    true_counts = []
    correctly_assigned = []
    incorrectly_assigned = []

    for i in range(k):
        #  find all data points that belong to class i
        idx = np.where(Class == i)[0]
        ysubgroup = y[idx]
        true_counts += [ysubgroup.shape[0]]

        unique, counts = np.unique(ysubgroup, return_counts=True)

        label = unique[np.argmax(counts)]
        labels += [label]

        correctly_assigned += [counts[np.argmax(counts)]]
        incorrectly_assigned += [ysubgroup.shape[0] - counts[np.argmax(counts)]]
        mismatch_rate = 1- counts[np.argmax(counts)] / ysubgroup.shape[0]
        mismatch_rates += [mismatch_rate]

    return labels, correctly_assigned, incorrectly_assigned, true_counts, mismatch_rates


