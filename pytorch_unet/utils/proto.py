from math import sqrt

import matplotlib.pyplot as plt
import tifffile as tiff
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision.transforms import Resize, ToTensor, Compose

from pytorch_unet.model.u_net import UNet
from pytorch_unet.processing.load import DataTransformer

torch.set_default_tensor_type('torch.FloatTensor')

data_path = "C:\\Users\\Mukesh\\Segmentation\\radnet\\data\\"

trainPath = data_path + 'train-volume.tif'
labelsPath = data_path + 'train-labels.tif'

# print("shape:", (tiff.imread(trainPath)).shape)
# print("Pil image 1: ", Image.fromarray((tiff.imread(trainPath))[1]))
# print("Pil image 2: ", Image.fromarray((tiff.imread(trainPath))[2]))
# print("Pil image 29: ", Image.fromarray((tiff.imread(trainPath))[29]))

train_data = Compose([Resize(64), ToTensor()])

# choose between augmentations for train data
# train_augment = augmentations()

train_transform = DataTransformer(trainPath, labelsPath, image_transform=train_data,
                                  image_augmentation=None)

# split the train and validation indices
train_indices, validation_indices = train_test_split(range(len(train_transform)), test_size=0.15)

# call the sampler for the train and validation data
train_samples = RandomSampler(train_indices)

# load train and validation data
train_loader = DataLoader(train_transform,
                          batch_size=1,
                          sampler=train_samples)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create the model
model = UNet(in_channels=1,
             n_classes=1,
             depth=3,
             wf=6,
             padding=True,
             batch_norm=False,
             up_mode='upsample').to(device)

# create the optimizer
optim = Adam(model.parameters())
loss_bce = nn.BCEWithLogitsLoss()
loss_values = []
epochs = 1
# start the training loop
for e in range(epochs):
    epoch_loss = 0.0

    for i, data in enumerate(train_loader):
        train_batch, train_labels = data
        train_batch, train_labels = train_batch.to(device), train_labels.to(device)

        # call the u-net module
        prediction = model(train_batch)

        # zero the parameter gradients
        optim.zero_grad()

        # call the loss function
        loss = loss_bce(prediction, train_labels)
        loss_values.append(loss)

        # backprop the loss
        loss.backward()
        optim.step()


