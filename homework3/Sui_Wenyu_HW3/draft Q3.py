# import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.neighbors import KernelDensity
import scipy.io as sio
from scipy import stats
from sklearn import preprocessing
import seaborn
from scipy.stats import multivariate_normal as mvn
import functions

# read data
label_mat = sio.loadmat('./data/label')
data_mat = sio.loadmat('./data/data')
true_label = label_mat['trueLabel'].flatten()
data = data_mat['data'].T


# Perform PCA.
# #Project the original data into 4-dimensional vectors
m, n = data.shape
mean = np.mean(data, axis = 0)
xc = data - mean

C = np.dot(xc.T, xc)/m
d = 4  # reduced dimension
values, V = np.linalg.eig(C)
ind = np.argsort(-values)[0:d]
V = V[:, ind]
values = values[ind]

# project the data to the top 4 principal directions
pdata = np.dot(xc,V) / np.sqrt(values)

# show the true distributions of 2 and 6 using first two PCA dimensions
plt.scatter(pdata[np.where(true_label == 2)[0], 0], pdata[np.where(true_label == 2)[0], 1], c = "#1f77b4", label = "2")
plt.scatter(pdata[np.where(true_label == 6)[0], 0], pdata[np.where(true_label == 6)[0], 1], c = '#ff7f0e', label = "6")
plt.title("Distributions of 2 and 6 using first two PCA dimensions")
plt.legend()
plt.show()



# perform EM algorithm MNIST data
# number of mixtures
K = 2

# initialize prior
pi = np.random.random(K)
pi = pi / np.sum(pi)

# initial mean and covariance
mu = np.random.randn(K, d)
mu_old = mu.copy()

sigma = []
for ii in range(K):
    # to ensure the covariance psd
    # np.random.seed(seed)
    dummy = np.random.randn(d, d)
    sigma.append(dummy @ dummy.T + np.identity(d))

# initialize the posterior
tau = np.full((m, K), fill_value=0.)

maxIter = 100
tol = 1e-3

plt.ion()

log_likelihood = []
for ii in range(maxIter):
    # E-step
    for kk in range(K):
        tau[:, kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])
    # normalize tau
    sum_tau = np.sum(tau, axis=1)
    sum_tau.shape = (m, 1)
    tau = np.divide(tau, np.tile(sum_tau, (1, K)))

    # M-step
    for kk in range(K):
        # update prior
        pi[kk] = np.sum(tau[:, kk]) / m

        # update component mean
        mu[kk] = pdata.T @ tau[:, kk] / np.sum(tau[:, kk], axis=0)

        # update cov matrix
        dummy = pdata - np.tile(mu[kk], (m, 1))  # X-mu
        sigma[kk] = dummy.T @ np.diag(tau[:, kk]) @ dummy / np.sum(tau[:, kk], axis=0)

    log_likelihood += [np.sum(np.log(sum_tau))]

    plt.figure(figsize=(6, 4))
    plt.scatter(pdata[:, 0], pdata[:, 1], c = tau[:, 0])
    plt.title('iteration '+ str(ii))
    plt.show()
    plt.pause(0.1)

    if np.linalg.norm(mu - mu_old) < tol:
        print('training coverged')
        break
    mu_old = mu.copy()
    if ii == 99:
        print('max iteration reached')
        break

plt.figure()
plt.plot(range(len(log_likelihood)), log_likelihood)
plt.title("log-likelihood function vs the number of iterations")
plt.show()



for kk in range(K):
    print("Wight of component ", kk+1, ": ", pi[kk])
    print("Mean of each variable of component ", kk+1, ": ")
    print(mu[kk, :])

    seaborn.heatmap(sigma[kk].real)
    plt.title('Intensity of Covariance Matrix of Component '+ str(kk+1))
    plt.show()


for kk in range(K):
    pixels = V @ np.diag(np.sqrt(values)) @ mu[kk,:].reshape((d,1)) + mean.reshape((-1,1))
    pixels = np.fliplr(pixels)
    functions.show_image(28, 28, pixels.real)


# determin the class that each data point belongs to
Class = np.argmax(tau, axis = 1)

# determines the true labels of each class and the mis-classification rate
assigned_class, correctly_assigned, true_counts, purities, mis_classification_rates = \
    functions.MNIST_true_labels_and_purity(Class, true_label)

EM_result = pd.DataFrame({'assigned class': assigned_class, \
                          'Correctly Assigned': correctly_assigned, \
                          "True Counts": true_counts, \
                          'purity': purities,\
                          'Mis Classification Rate': mis_classification_rates})

print("Mis-classification Rate and Assigned Label of each Component - EM Algorithm ")
display(EM_result)

print("Overall Mis-classification Rate - EM Algorithm: ")
print(round(1 - np.sum(correctly_assigned) / np.sum(true_counts),4))


# implement K_means clustering
Class_kmeans, xcentroid = (functions.k_means_MNIST_l2(data,2))

#functions.show_image(28, 28, xcentroid[0, :])
#functions.show_image(28, 28, xcentroid[1, :])


# determines the true labels of each class and the mis-classification rate
assigned_class, correctly_assigned, true_counts, purities, mis_classification_rates = \
    functions.MNIST_true_labels_and_purity(Class_kmeans, true_label)

kmeans_result = pd.DataFrame({'assigned class': assigned_class, \
                          'Correctly Assigned': correctly_assigned, \
                          "True Counts": true_counts, \
                          'purity': purities,\
                          'Mis Classification Rate': mis_classification_rates})

print("Mis-classification Rate and Assigned Label of each Component - K means Clustering ")
display(kmeans_result)

print("Overall Mis-classification Rate - K means clustering: ")
print(round(1 - np.sum(correctly_assigned) / np.sum(true_counts),4))