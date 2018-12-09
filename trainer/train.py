from preprocessing.load import DataTransformer
from model.u_net import UNet
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import os
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.optim import Adam
from preprocessing.augments import augmentations

from sklearn.model_selection import train_test_split
import numpy as np

import tifffile as tiff
from torchsummary import summary

################################################################
# setting up a logger
import logging

logging.basicConfig(filename='train.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

################################################################

# CUDA for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    """ Contains the training loop"""

    # get data set file path
    data_path = os.path.join(args.root_dir, 'data', 'train-volume.tif')
    labels_path = os.path.join(args.root_dir, 'data', 'train-labels.tif')

    # compose the transforms for the train set
    train_data = Compose([Resize(args.image_size), ToTensor()])

    # choose between augmentations for train data
    if args.augment == 'yes':
        train_augment = augmentations()
        train_transform = DataTransformer(data_path, labels_path,
                                          image_transform=train_data, image_augmentation=train_augment)

    elif args.augment == 'no':
        # transforming the train data and returning a 4D tensor
        train_transform = DataTransformer(data_path, labels_path,
                                          image_transform=train_data, image_augmentation=None)

    # transform for validation data
    val_data = Compose([Resize(args.image_size), ToTensor()])
    val_transform = DataTransformer(data_path, labels_path,
                                    image_transform=val_data, image_augmentation=None)

    # split the train and validation indices
    train_indices, validation_indices = train_test_split(range(len(train_transform)), test_size=0.15)

    # call the sampler for the train and validation data
    train_samples = RandomSampler(train_indices)
    validation_samples = SequentialSampler(validation_indices)

    # load train and validation data
    train_loader = DataLoader(train_transform,
                              batch_size=args.batch_size,
                              sampler=train_samples)

    val_loader = DataLoader(val_transform,
                            batch_size=args.batch_size,
                            sampler=validation_samples)

    # create the model
    model = UNet(in_channels=1,
                 n_classes=args.n_classes,
                 depth=args.depth,
                 wf=6,
                 padding=False,
                 batch_norm=False,
                 up_mode=args.up_mode).to(device)

    # create the optimizer
    optim = Adam(model.parameters())

    # start the training loop
    for e in args.epochs:
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
