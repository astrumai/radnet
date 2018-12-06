from preprocessing.load import DataTransformer, sampler
from preprocessing.augments import augmentations
from model.u_net import UNet
import os
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.optim import Adam
from sklearn.model_selection import train_test_split

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


def split_data(dataset, args):
    """A function to split the data
    Arguments:
        dataset:
        args:
    Returns:
        List containing train-test split of inputs.
    """
    train_data, val_data = train_test_split(dataset, test_size=args.test_size, shuffle=True)

    return train_data, val_data


def train(args):
    """ Contains the training loop"""

    # get data set file path
    train_path = os.path.join(args.root_dir, 'data', 'train-volume.tif')
    labels_path = os.path.join(args.root_dir, 'data', 'train-labels.tif')

    # transforming the train data and returning a 4D tensor
    data = Compose([Resize(args.image_size), ToTensor()])
    data_transform = DataTransformer(train_path, labels_path, data)

    # split the data into train and validation set
    train_data, val_data = split_data(data_transform, args)

    # choose between augmentations for train data
    if args.augment == 'yes':
        # calling the augmentations
        train_augment = augmentations(train_data, args)
        train_samples = sampler(train_augment)

    elif args.augment == 'no':
        # transforming the train data and returning a 4D tensor
        train_samples = sampler(train_data)

    # use the random sampler to sample train and validation data into the train data loader
    val_samples = sampler(val_data)

    # load train and validation data
    train_loader = DataLoader(train_augment,
                              args.batch_size,
                              sampler=train_samples)

    val_loader = DataLoader(val_data,
                            args.batch_size,
                            sampler=val_samples)

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
