import torch
from PIL import Image
import tifffile as tiff
from torchvision.transforms import *
from preprocessing.load import DataTransformer
from preprocessing.augments import augmentations
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam
from model.u_net import UNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import RandomSampler

torch.set_default_tensor_type('torch.FloatTensor')

data_path = "C:\\Users\\Mukesh\\Segmentation\\U-net\\data\\"

trainPath = data_path + 'train-volume.tif'
labelsPath = data_path + 'train-labels.tif'

# print("shape:", (tiff.imread(trainPath)).shape)
# print("Pil image 1: ", Image.fromarray((tiff.imread(trainPath))[1]))
# print("Pil image 2: ", Image.fromarray((tiff.imread(trainPath))[2]))
# print("Pil image 29: ", Image.fromarray((tiff.imread(trainPath))[29]))

train_data = Compose([Resize(64), ToTensor()])
# temp = [transform(Image.fromarray((tiff.imread(trainPath))[1]))]
# print("Transform for image 1", temp)

# choose between augmentations for train data
train_augment = augmentations()
train_transform = DataTransformer(trainPath, labelsPath,
                                  image_transform=train_data, image_augmentation=train_augment)

# split the train and validation indices
train_indices, validation_indices = train_test_split(range(len(train_transform)), test_size=0.15)

# call the sampler for the train and validation data
train_samples = RandomSampler(train_indices)

# load train and validation data
train_loader = DataLoader(train_transform,
                          batch_size=4,
                          sampler=train_samples)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create the model
model = UNet(in_channels=1,
             n_classes=2,
             depth=5,
             wf=6,
             padding=False,
             batch_norm=False,
             up_mode='upsample').to(device)

# create the optimizer
optim = Adam(model.parameters())
epochs = 5
# start the training loop
for e in range(epochs):
    epoch_loss = 0.0

    for i, data in enumerate(train_loader):
        train_batch, train_labels = data
        train_batch, train_labels = train_batch.to(device), train_labels.to(device)

        # call the u-net module
        print(model)
        prediction = model(train_batch)

        # zero the parameter gradients
        optim.zero_grad()

        # call the loss function
        loss = F.cross_entropy(prediction, train_labels)

        # backprop the loss
        loss.backward()
        optim.step()

        # print the loss
        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(e,
                                                           i,
                                                           len(train_loader),
                                                           loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(e,
                                                             epoch_loss / len(train_loader)))
