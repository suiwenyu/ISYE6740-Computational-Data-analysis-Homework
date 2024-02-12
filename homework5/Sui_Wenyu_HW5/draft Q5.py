import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error


# read data
matFile = sio.loadmat('./data/cs.mat')
x = matFile['img']

# Show original image
plt.figure()
plt.imshow(x, cmap = 'gray')
plt.title("True Image")
plt.show()

x= np.reshape(x, (2500,))

# generate matrix A, epsilon and y
A = np.random.randn(1300, 2500)
eps = np.random.randn(1300) * 5
y = np.dot(A, x) + eps


"""
# LASSO cross validation
clf = LassoCV(cv = 10).fit(A,y)

mse_path = clf.mse_path_
alphas = clf.alphas_
best_alpha = clf.alpha_
print("Best Alpha value chosen by 10 fold cross validation: ", round(best_alpha,5))

mean_mse_path = np.mean(mse_path, axis = 1)

# display cv error curve, represented by mean MSE
plt.figure()
plt.plot(alphas, mean_mse_path)
plt.xlabel("Alpha")
plt.ylabel("Average Mean Squared Error (MSE)")
plt.title("CV Error Curve (represented by average MSE)")
plt.show()



# retrieve LASSO coefficient and reconstruct the image
importance = clf.coef_
reconstruct = np.reshape(importance, (50,50))

plt.figure()
plt.imshow(reconstruct, cmap = "gray")
plt.title("Image Reconstructed with LASSO Coefficients")
plt.show()
"""

# Ridge cross validation
cv_split = KFold(n_splits = 10, shuffle = True)


ridge = Ridge(positive= True)
alphas = np.arange(1,200,1)
para = {'alpha': alphas}

cv_search = GridSearchCV(estimator = ridge,
                         scoring = 'neg_mean_squared_error',
                         cv = cv_split,
                         param_grid = para,
                         ).fit(A, y)

mean_mse_path = -cv_search.cv_results_['mean_test_score']

plt.figure()
plt.plot(alphas, mean_mse_path)
plt.show()
print("end")



"""
lasso = Lasso(max_iter=10000)
alphas = np.arange(0.01,2,0.005)
para = {'alpha': alphas}

cv_search = GridSearchCV(estimator = lasso,
                         scoring = 'neg_mean_squared_error',
                         cv = cv_split,
                         param_grid = para).fit(A, y)

mean_mse_path = -cv_search.cv_results_['mean_test_score']

plt.figure()
plt.plot(alphas, mean_mse_path)
plt.show()
print(mean_mse_path)
"""