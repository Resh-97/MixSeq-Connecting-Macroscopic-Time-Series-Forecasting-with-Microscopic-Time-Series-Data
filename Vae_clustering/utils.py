from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import pandas as pd



def plot_clustering(z_run, download = False, folder_name ='clustering'):

    z_run_pca = TruncatedSVD(n_components=2).fit_transform(z_run)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(z_run_pca)
    label = kmeans.labels_
    hex_colors = []
    for value in np.unique(label):
        if value == 1:
            hex_colors.append('#00FF00')
        elif value == 2:
            hex_colors.append('#FF0000')
        elif value == 3:
            hex_colors.append('#0000FF')
        else:
             hex_colors.append('#000000')

    colors = [hex_colors[int(i)] for i in label]

    plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=colors, marker='*', linewidths=0)
    plt.title('Clustered Latent vectors after PCA')
    if download:
        if os.path.exists(folder_name):
            pass
        else:
            os.mkdir(folder_name)
        plt.savefig(folder_name + "/clustered_data_4.png")
    else:
        plt.show()

def open_data(direc, ratio_train=0.8, dataset="syntheticARMA_4"):

    train_set = pd.read_csv(direc + '/' + dataset +".csv",index_col=0)
    data = np.expand_dims(train_set.iloc[: , :], -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 2:, :], data[ind[ind_cut:], 2:, :]
