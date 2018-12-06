import tifffile as tiff
import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataset import Dataset
from PIL import Image


class DataTransformer(Dataset):
    """ Dataset loader to pass to the pytorch DataLoader
    Arguments:
          train_filename :
          labels_filename:
    Returns:
        a list of train_images, train_labels 4D tensor tuples
    """

    def __init__(self,
                 train_filename,
                 labels_filename,
                 transform=None):

        self.train_filename = train_filename
        self.labels_filename = labels_filename
        self.transform = transform
        self.len_train = tiff.imread(self.train_filename).shape[0]

    def __len__(self):
        return self.len_train

    def _read_data(self, index):
        return Image.fromarray((tiff.imread(self.train_filename))[index])

    def _read_labels(self, index):
        return Image.fromarray((tiff.imread(self.labels_filename))[index])

    def __getitem__(self, index):
        images = self.transform(self._read_data(index))
        labels = self.transform(self._read_labels(index))

        return [images, labels]


def sampler(data):
    data_sampler = RandomSampler(data)
    data_samples = next(iter(data_sampler))

    return data_samples

