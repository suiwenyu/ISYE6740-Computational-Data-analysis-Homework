# import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.neighbors import KernelDensity
import seaborn as sns
import functions



# read data
n90pol = pd.read_csv(f'data/n90pol.csv')
amygdala = np.array(n90pol['amygdala'])
acc = np.array(n90pol['acc'])

n90data = np.array(n90pol)[:, 0:2]
n90y = np.array(n90pol)[:, 2]


stdev = np.std(amygdala)
h = 1.06 * stdev / np.power(amygdala.shape[0], 0.2)

stdev2 = np.std(acc)
h2 = 1.06 * stdev2 / np.power(acc.shape[0], 0.2)

#functions. TwoD_histogram(data = n90data, nbin = 20,\
 #                         xlabel = n90pol.columns[0], \
  #                        ylabel = n90pol.columns[1])




h_avg = (h + h2) / 2

# display joint conditional KDEs of amygdala and acc when orientation = 2,3,4,5,6
for i in range(2, 6):
    fig = plt.figure(figsize=(11, 3))
    ax = plt.subplot(1, 2, 1, projection='3d')
    # ax = fig.add_subplot(111, projection='3d')

    idx = np.where(n90y == i)[0]
    n90data_sub = n90data[idx, :]

    x = np.arange(-0.1, 0.12, 0.001)
    xpos, ypos = np.meshgrid(x, x)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    pos = np.concatenate((xpos.reshape(-1, 1), ypos.reshape(-1, 1)), axis=1)

    kde_joint = KernelDensity(kernel='gaussian', bandwidth=h_avg).fit(n90data_sub)
    log_density_joint = kde_joint.score_samples(pos)
    density_joint = np.exp(log_density_joint)

    zpos = np.zeros_like(xpos)

    ax.bar3d(xpos, ypos, zpos, 0.001, 0.001, density_joint, cmap="Blues")

    plt.title("2-dimensional KDE - 3D histogram")

    # 2-dimensional KDE in heatmap
    ax = plt.subplot(1, 2, 2)
    zz = density_joint.reshape(x.shape[0], x.shape[0])
    ax.pcolormesh(x, x, zz, shading='auto', cmap="Reds")
    plt.scatter(n90data_sub[:, 0], n90data_sub[:, 1], cmap="Blues", s=6)
    plt.title("2-dimensional KDE - heatmap, oritentation = " + str(i))
    plt.show()