import tifffile as tiff
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np


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


