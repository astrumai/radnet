import os

import numpy as np
import tifffile as tiff
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, ToTensor

from pytorch_unet.processing.augments import augmentations


class DataTransformer(Dataset):
    """ Dataset loader to pass to the pytorch DataLoader
    Arguments:
          train_filename :
          labels_filename:
    Returns:
        a list of train_images, train_labels 4D tensor tuples
    """

    def __init__(self, train_filename, labels_filename, image_transform=None, image_augmentation=None):
        self.train_filename = train_filename
        self.labels_filename = labels_filename
        self.image_transform = image_transform
        self.image_augmentation = image_augmentation
        self.len_train = tiff.imread(self.train_filename).shape[0]

    def __len__(self):
        return self.len_train

    def _read_data(self, index):
        return Image.fromarray((tiff.imread(self.train_filename))[index])

    def _read_labels(self, index):
        return Image.fromarray((tiff.imread(self.labels_filename))[index])

    def __getitem__(self, index):
        if self.labels_filename is not None:
            images = self._read_data(index)
            labels = self._read_labels(index)

            if self.image_augmentation is not None:
                x = np.array(images)
                y = np.array(labels)
                data = {'input': x, 'mask': y}
                aug_data = self.image_augmentation(data)
                trans_images = aug_data['input']
                trans_labels = aug_data['mask']

            if self.image_augmentation is None:
                trans_images = self.image_transform(images)
                trans_labels = self.image_transform(labels)
            return [trans_images, trans_labels]

        if self.labels_filename is None:
            images = self._read_data(index)
            trans_images = self.image_transform(images)
            return trans_images


def load_data(args):
    """Load data from here and return train loader and validation loader"""
    # get data set file path
    data_path = os.path.join(args.main_dir, 'data', 'train-volume.tif')
    labels_path = os.path.join(args.main_dir, 'data', 'train-labels.tif')

    # compose the transforms for the train set
    train_data = Compose([Resize(args.image_size), ToTensor()])

    # choose between augmentations for train data
    if args.augment == 'yes':
        train_augment = augmentations(args)
        train_transform = DataTransformer(data_path, labels_path, image_transform=train_data,
                                          image_augmentation=train_augment)

    elif args.augment == 'no':
        # transforming the train data and returning a 4D tensor
        train_transform = DataTransformer(data_path, labels_path, image_transform=train_data, image_augmentation=None)

    # transform for validation data
    val_data = Compose([Resize(args.image_size), ToTensor()])
    val_transform = DataTransformer(data_path, labels_path, image_transform=val_data, image_augmentation=None)

    # split the train and validation indices
    train_indices, validation_indices = train_test_split(range(len(train_transform)), test_size=0.15)

    # call the sampler for the train and validation data
    train_samples = RandomSampler(train_indices)
    validation_samples = SequentialSampler(validation_indices)

    # load train and validation data
    train_loader = DataLoader(train_transform, batch_size=args.batch_size, sampler=train_samples)
    val_loader = DataLoader(val_transform, batch_size=args.batch_size, sampler=validation_samples)

    return train_loader, val_loader
