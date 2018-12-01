import tifffile as tiff
from torch.utils.data.dataset import Dataset
from PIL import Image


class SegmentationLoader(Dataset):
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

    def read_train(self, index):
        return Image.fromarray((tiff.imread(self.train_filename))[index])

    def read_labels(self, index):
        return Image.fromarray((tiff.imread(self.labels_filename))[index])

    def __getitem__(self, index):
        train_images = self.transform(self.read_train(index))
        train_labels = self.transform(self.read_labels(index))

        return [train_images, train_labels]

