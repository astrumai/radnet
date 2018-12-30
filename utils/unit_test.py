import os
import unittest

import tifffile as tiff
from PIL import Image

from processing.augments import *
from utils.metrics import dice
from ..processing.load import DataTransformer


class TestUNet(unittest.TestCase):
    def test_data_transformer(self):
        """Test for data loader"""

        root_dir = "C:\\Users\\Mukesh\\Segmentation\\U-net\\"
        # train_path = os.path.join(args.root_dir, 'data', 'train-volume.tif')
        # labels_path = os.path.join(args.root_dir, 'data', 'train-labels.tif')
        train_path = os.path.join(root_dir, 'data', 'train-volume.tif')
        labels_path = os.path.join(root_dir, 'data', 'train-labels.tif')

        # transform = Compose([Resize(args.image_size), ToTensor()])
        transform = Compose([Resize(64), ToTensor()])
        dataset = DataTransformer(train_path, labels_path, transform)
        temp = [transform(Image.fromarray((tiff.imread(train_path))[1]))]

        self.assertEqual(torch.all(torch.eq((dataset.__getitem__(1)[0]), (temp[0]))), msg="The Tensors are not equal")

    def test_unet(self):
        """Test U-Net model"""

        from model.u_net import UNet

        model = UNet(in_channels=1,
                     n_classes=2,
                     depth=3,
                     wf=6,
                     padding=False,
                     batch_norm=False)
        print(model)

    def test_dice(self):
        image1 = [0, 1, 0]
        image2 = [0, 1, 0]
        true_coeff = 1.0
        dice_coeff = dice(image1, image2)
        self.assertEqual(true_coeff, dice_coeff, msg="Dice Test failed")


class TestAugmentations(unittest.TestCase):
    """Test if you are getting correct augmentations"""
    pass


if __name__ == '__main__':
    unittest.main()
