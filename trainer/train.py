from preprocessing.load import SegmentationLoader
import os
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

################################################################
# setting up a logger
import logging

logging.basicConfig(filename='train.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


################################################################


def train(args):
    """ Contains the training loop"""

    # get data set file path
    train_path = os.path.join(args.root_dir, 'data', 'train-volume.tif')
    labels_path = os.path.join(args.root_dir, 'data', 'train-labels.tif')

    # transform the training data and return a 4D tensor
    transform = Compose([Resize(args.image_size), ToTensor()])

    # calling the SegmentationLoader
    train_data = SegmentationLoader(train_path, labels_path, transform)

    # load train data
    # what is the output of a data loader?
    train_loader = DataLoader(train_data,
                              args.batch_size,
                              shuffle=True)

    # start the training loop
    for e in args.epochs:
        # create the u-net module

        # call the training function to initiate training

        # call the evaluation function for predictions

        pass

    pass
