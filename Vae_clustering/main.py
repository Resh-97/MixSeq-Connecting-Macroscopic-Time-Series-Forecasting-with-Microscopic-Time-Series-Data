from vrae import VRAE
from utils import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


hidden_size = 90
hidden_layer_depth = 1
latent_length = 20
batch_size = 32
learning_rate = 0.0005
n_epochs = 40
dropout_rate = 0.2
optimizer = 'Adam'
cuda = True
print_every = 30
clip = True
max_grad_norm = 5
loss = 'MSELoss'
block = 'LSTM'
dload = './model_dir'

X_train, X_val = open_data('data', ratio_train=0.9)

train_dataset = TensorDataset(torch.from_numpy(X_train))
test_dataset = TensorDataset(torch.from_numpy(X_val))

sequence_length = X_train.shape[1]
number_of_features = X_train.shape[2]

vrae = VRAE(sequence_length=sequence_length,
            number_of_features=number_of_features,
            hidden_size=hidden_size,
            hidden_layer_depth=hidden_layer_depth,
            latent_length=latent_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            cuda=cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss=loss,
            block=block,
            dload=dload)

#vrae.fit(train_dataset)

# To load a presaved model, execute:
vrae.load('model_dir\\vrae_4.pth')
z_run = vrae.transform(test_dataset)

#To save latent vector, pass the parameter `save`
# z_run = vrae.transform(dataset, save = True)
#vrae.save('vrae_3.pth')

plot_clustering(z_run, download=True)
print("The end")
