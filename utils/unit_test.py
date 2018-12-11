import unittest
from processing.augments import *
import numpy as np
from processing.augments import VerticalFlip
import os
from processing.load import DataTransformer
from torchvision.transforms import Compose, Resize, ToTensor
from utils.helpers import convert_2d_to_target
from utils.metrics import dice
import torch
from PIL import Image
import tifffile as tiff
import pytest


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

    def test_clip(self):
        img = np.array(
            [[-300, 0],
             [100, 400]], dtype=np.float32)
        expected = np.array(
            [[0, 0],
             [100, 255]], dtype=np.float32)
        clipped = clip(img, dtype=np.uint8, maxval=255)
        self.assertEqual(clipped, expected, msg="Clip Test Failed")

    @pytest.mark.parameterize('target', ['image', 'mask'])
    def test_vertical_flip(self, target):
        img = np.array(
            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1]], dtype=np.uint8)
        expected = np.array(
            [[0, 0, 1],
             [0, 1, 1],
             [1, 1, 1]], dtype=np.uint8)
        img, expected = convert_2d_to_target([img, expected], target=target)
        flipped_img = VerticalFlip(img)
        self.assertEqual(flipped_img, expected, msg="Vertical Flip Test Failed")

    @pytest.mark.parameterize('target', ['image', 'mask'])
    def test_horizontal_flip(self, target):
        img = np.array(
            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1]], dtype=np.uint8)
        expected = np.array(
            [[1, 1, 1],
             [1, 1, 0],
             [1, 0, 0]], dtype=np.uint8)
        img, expected = convert_2d_to_target([img, expected], target=target)
        flipped_img = HorizontalFlip(img)
        self.assertEqual(flipped_img, expected, msg="Horizontal Flip Test Failed")

    @pytest.mark.parameterize('target', ['image', 'mask'])
    def test_rotate90(self, target):
        img = np.array(
            [[0, 0, 1],
             [0, 0, 1],
             [0, 0, 1]], dtype=np.uint8)
        expected = np.array(
            [[1, 1, 1],
             [0, 0, 0],
             [0, 0, 0]], dtype=np.uint8)
        img, expected = convert_2d_to_target([img, expected], target=target)
        rotated_img = Rotate90(img)
        self.assertEqual(rotated_img, expected, msg="Rotation Test Failed")

    @pytest.mark.parameterize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    def test_elastic_transform(self, monkeypatch, interpolation):
        image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
        monkeypatch.setattr()
        aug = ElasticTransform(alpha=1, sigma=50, alpha_affine=50,
                               interpolation=interpolation)
        data = aug(image=image, mask=mask)
        expected_image = elastic_transform_fast(image, alpha=1, sigma=50,
                                                alpha_affine=50, interpolation=interpolation,
                                                border_mode=cv2.BORDER_REFLECT_101,
                                                random_state=np.random.RandomState(1111))
        expected_mask = elastic_transform_fast(mask, alpha=1, sigma=50, alpha_affine=50,
                                               interpolation=cv2.INTER_NEAREST,
                                               border_mode=cv2.BORDER_REFLECT_101,
                                               random_state=np.random.RandomState(1111))
        self.assertEqual(data['image'], expected_image, msg="The images are not elastically transformed properly")
        self.assertEqual(data['mask'], expected_mask, msg="The masks are not elastically transformed properly")

    @pytest.mark.parameterize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    def test_grid_distortion(self, interpolation):
        image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
        aug = GridDistortion(num_steps=1, distort_limit=(0.3, 0.3), interpolation=interpolation)
        data = aug(image=image, mask=mask)
        expected_image = grid_distortion(image, num_steps=1, xsteps=[1.3], ysteps=[1.3], interpolation=interpolation,
                                         border_mode=cv2.BORDER_REFLECT_101)
        expected_mask = grid_distortion(mask, num_steps=1, xsteps=[1.3], ysteps=[1.3], interpolation=cv2.INTER_NEAREST,
                                        border_mode=cv2.BORDER_REFLECT_101)
        self.assertEqual(data['image'], expected_image, msg="The images are not grid distorted properly")
        self.assertEqual(data['mask'], expected_mask, msg="The masks are not grid distorted properly")

    # @pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    def test_optical_distortion(self):
        interpolation = cv2.INTER_NEAREST
        image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
        aug = OpticalDistortion(distort_limit=(0.05, 0.05), shift_limit=(0, 0), interpolation=interpolation)
        data = aug(image=image, mask=mask)
        expected_image = optical_distortion(image, k=0.05, dx=0, dy=0, interpolation=interpolation,
                                            border_mode=cv2.BORDER_REFLECT_101)
        expected_mask = optical_distortion(mask, k=0.05, dx=0, dy=0, interpolation=cv2.INTER_NEAREST,
                                           border_mode=cv2.BORDER_REFLECT_101)
        self.assertEqual(data['image'], expected_image, msg="The images are not optically distorted properly")
        self.assertEqual(data['mask'], expected_mask, msg="The masks are not optically distorted properly")

    # @pytest.mark.parameterize(['beta', 'expected'], [(0.2, 0.48), (-0.1, 0.36)])
    def test_brightness(self):
        beta = 0.2
        expected = -0.1
        img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
        expected = np.ones_like(img) * expected
        img = brightness_contrast_adjust(img, beta=beta)
        assert (img.dtype == np.dtype('float32'))
        self.assertAlmostEqual(img, expected, msg="Brightness Test Failed")

    # @pytest.mark.parametrize(['alpha', 'expected'], [(1.5, 0.6), (3, 1.0)])
    def test_contrast(self):
        alpha = 1.5
        expected = 3
        img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
        expected = np.ones((100, 100, 3), dtype=np.float32) * expected
        img = brightness_contrast_adjust(img, alpha=alpha)
        assert (img.dtype == np.dtype('float32'))
        self.assertAlmostEqual(img, expected, msg="Contrast Test Failed")

    # @pytest.mark.parameterize(['gamma', 'expected'], [(1, 0.4), (10, 0.00010486)])
    def test_gamma_change(self):
        gamma = 1
        expected = 10
        img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
        expected = np.ones((100, 100, 3), dtype=np.float32) * expected
        img = gamma_transform(img, gamma=gamma)
        assert img.dtype == np.dtype('float32')
        assert np.allclose(img, expected)


if __name__ == '__main__':
    unittest.main()
