from matplotlib import pyplot as plt
from vrae import VRAE
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

#NOTE: currently this assumes that the latent space is outputted in the same order as the data is inputted
def open_data(direc, ratio_train=0.8, dataset=None):

    train_set = pd.read_csv(direc + '/' + dataset +".csv",index_col=0)
    data = np.expand_dims(train_set.iloc[: , :], -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)

    Xtrain_f = data[ind[:ind_cut], :, :]
    Xval_f = data[ind[ind_cut:], :, :]

    Xtrain_labels = Xtrain_f[:,0,0]
    Xval_labels = Xval_f[:,0,0]

    Xtrain = Xtrain_f[:,1:,:]
    Xval = Xval_f[:, 1:, :]

    return Xtrain_labels, Xval_labels, Xtrain, Xval

def plot_clustering(k, z_run, download = False, folder_name ='clustering'):

    z_run_pca = TruncatedSVD(n_components=2).fit_transform(z_run)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(z_run_pca)
    label = kmeans.labels_
    hex_colors = []
    for value in np.unique(label):
        if value == 0:
            hex_colors.append('#FF0000')
        if value == 1:
            hex_colors.append('#FFFF00')
        if value == 2:
            hex_colors.append('#00FF00')
        else:
             hex_colors.append('#0000FF')

    colors = [hex_colors[int(i)] for i in label]

    plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=colors, marker='*', linewidths=0)
    plt.title('Clustered Latent vectors after PCA')
    if download:
        if os.path.exists(folder_name):
            pass
        else:
            os.mkdir(folder_name)
        plt.savefig(folder_name + "/clustered_data.png")
    else:
        plt.show()

    return label

#load all the datasets
X_train_labels2, X_val_labels2, X_train_2, X_val_2 = open_data('data', ratio_train=0.9, dataset="syntheticARMA_2")
X_train_labels3, X_val_labels3, X_train_3, X_val_3 = open_data('data', ratio_train=0.9, dataset="syntheticARMA_3")
X_train_labels4, X_val_labels4, X_train_4, X_val_4 = open_data('data', ratio_train=0.9, dataset="syntheticARMA_4")
X_train_labels5, X_val_labels5, X_train_5, X_val_5 = open_data('data', ratio_train=0.9, dataset="syntheticARMA_5")

train_dataset_2 = TensorDataset(torch.from_numpy(X_train_2))
test_dataset_2 = TensorDataset(torch.from_numpy(X_val_2))

train_dataset_3 = TensorDataset(torch.from_numpy(X_train_3))
test_dataset_3 = TensorDataset(torch.from_numpy(X_val_3))

train_dataset_4 = TensorDataset(torch.from_numpy(X_train_4))
test_dataset_4 = TensorDataset(torch.from_numpy(X_val_4))

train_dataset_5 = TensorDataset(torch.from_numpy(X_train_5))
test_dataset_5 = TensorDataset(torch.from_numpy(X_val_5))

sequence_length = X_train_3.shape[1]
number_of_features = X_train_3.shape[2]

#define the models
hidden_size = 90
hidden_layer_depth = 1
latent_length = 20
batch_size = 32
learning_rate = 0.0005
n_epochs = 40
dropout_rate = 0.2
optimizer = 'Adam'
cuda = True
print_every=30
clip = True
max_grad_norm=5
loss = 'MSELoss'
block = 'LSTM'
dload = './model_dir'

vrae_2 = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload)

vrae_3 = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload)

vrae_4 = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload)

vrae_5 = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload)

vrae_2.load('model_dir/vrae_2.pth')
vrae_3.load('model_dir/vrae_3.pth')
vrae_4.load('model_dir/vrae_4.pth')
vrae_5.load('model_dir/vrae_5.pth')

#produce the latent space using the models
z_run_2 = vrae_2.transform(test_dataset_2)
z_run_3 = vrae_3.transform(test_dataset_3)
z_run_4 = vrae_4.transform(test_dataset_4)
z_run_5 = vrae_4.transform(test_dataset_5)

#cluster the latent space values using the models
labels_2 = plot_clustering(2, z_run_5, download = False)
labels_3 = plot_clustering(3, z_run_5, download = False)
labels_4 = plot_clustering(4, z_run_5, download = False)
labels_5 = plot_clustering(5, z_run_5, download = False)
labels_6 = plot_clustering(6, z_run_5, download = False)
labels_7 = plot_clustering(7, z_run_5, download = False)
#plt.show()

#compare to the original dataset cluster labels for accuracy

#define function
def calculate_accuracy(true_labels, predicted_labels, k, zrun):
    #distortion?
    distortion_total = 0
    for i in range(k):
        try:
            zrun = TruncatedSVD(n_components=2).fit_transform(zrun)
            pred_dist = zrun[predicted_labels == i, :]
            mean = np.sum(pred_dist, 0)/pred_dist.shape[0]
            distortion = np.linalg.norm(pred_dist-mean)
            distortion_total += distortion
        except:
            distortion_total += 0
    distortion = distortion_total/k

    #Accuracy Code
    # #F score later???
    # fin_accuracy = []
    # best_acc = 0
    # for shift in range(k):
    #     accuracy = []
    #     total_acc = 0
    #     for i in range(k):
    #         #K-means might class '0' as '1' and vice versa
    #         preds = predicted_labels[predicted_labels==i]
    #         preds = preds+(shift*np.ones(preds.shape))
    #         preds = (preds % k)+1
    #
    #         truths = true_labels[predicted_labels==i]
    #
    #         total_acc += (sum(preds==truths)/(len(truths)+1E-100))
    #         accuracy.append(sum(preds==truths)/(len(truths)+1E-100))
    #
    #     if total_acc > best_acc:
    #         best_acc = total_acc
    #         fin_accuracy = accuracy


    return distortion

cluster_acc_2 = calculate_accuracy(X_val_labels2, labels_2, 2, z_run_5)
cluster_acc_3 = calculate_accuracy(X_val_labels3, labels_3, 3, z_run_5)
cluster_acc_4 = calculate_accuracy(X_val_labels5, labels_4, 4, z_run_5)
cluster_acc_5 = calculate_accuracy(X_val_labels5, labels_5, 5, z_run_5)
cluster_acc_6 = calculate_accuracy(X_val_labels4, labels_6, 6, z_run_5)
cluster_acc_7 = calculate_accuracy(X_val_labels5, labels_7, 7, z_run_5)

#cluster_acc_6 = calculate_accuracy(X_val_labels5+np.ones(X_val_labels5.shape), labels_5+np.ones(X_val_labels5.shape), 6)
#cluster_acc_7 = calculate_accuracy(X_val_labels5+2*np.ones(X_val_labels5.shape), labels_5+2*np.ones(X_val_labels5.shape), 7)

print(cluster_acc_2)
print(cluster_acc_3)
print(cluster_acc_4)
print(cluster_acc_5)


# avg_acc_2 = sum(cluster_acc_2)/len(cluster_acc_2)
# avg_acc_3 = sum(cluster_acc_3)/len(cluster_acc_3)
# avg_acc_4 = sum(cluster_acc_4)/len(cluster_acc_4)
# avg_acc_5 = sum(cluster_acc_5)/len(cluster_acc_5)
# avg_acc_6 = sum(cluster_acc_6)/len(cluster_acc_6)
# avg_acc_7 = sum(cluster_acc_7)/len(cluster_acc_7)

avg_acc_2 = cluster_acc_2
avg_acc_3 = cluster_acc_3
avg_acc_4 = cluster_acc_4
avg_acc_5 = cluster_acc_5
avg_acc_6 = cluster_acc_6
avg_acc_7 = cluster_acc_7


#plot graph
plt.plot([2, 3, 4, 5, 6, 7], [avg_acc_2, avg_acc_3, avg_acc_4, avg_acc_5, avg_acc_6, avg_acc_7], 'o-')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.title('Number of Clusters against Distortion on Synthetic ARMA Dataset')
plt.show()