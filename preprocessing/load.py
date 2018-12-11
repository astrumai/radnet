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
        images = self._read_data(index)
        labels = self._read_labels(index)

        if self.image_augmentation is not None:
            print("Aug")
            image_array = np.array(images)
            label_array = np.array(labels)
            aug_images = self.image_augmentation(image=image_array, mask=label_array)
            trans_images = Image.fromarray(aug_images["image"])
            trans_labels = Image.fromarray(aug_images["mask"])
        if self.image_augmentation is None:
            trans_images = self.image_transform(images)
            trans_labels = self.image_transform(labels)
        return [trans_images, trans_labels]

