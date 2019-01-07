import random

import cv2
import numpy as np
import torch
from torchvision.transforms import RandomApply, Compose


class PrepareImageAndMask(object):
    """Prepare images and masks like fixing channel numbers."""

    def __call__(self, data):
        img = data['input']
        img = img[:, :, :3]  # max 3 channels
        img = img / 255

        if 'mask' in data:
            mask = data['mask']
        else:
            mask = np.zeros(img.shape[:2], dtype=img.dtype)

        data['input'] = img.astype(np.float32)
        data['mask'] = mask.astype(np.float32)
        return data


def to_tensor(pic):
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((0, 1, 2)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class ConvertToTensor(object):
    """ Converts the image to tensor.
    Note:
        Modified from PyTorch vision ToTensor. Converts a PIL Image or numpy.ndarray (H x W x C) in the
        range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] by calling the
        to_tensor function.
    """
    def __call__(self, data):
        trans_images_arr = np.expand_dims(data['input'], axis=0)
        trans_labels_arr = np.expand_dims(data['mask'], axis=0)
        data['input'] = to_tensor(trans_images_arr)
        data['mask'] = to_tensor(trans_labels_arr)
        return data


class ResizeToNxN(object):
    """Resize input images to rgb NxN and the masks into gray NxN.
     Note:
         uses cv2.INTER_LINEAR which implements bilinear interpolation for resizing.
    """

    def __init__(self, n=128):
        self.n = n

    def __call__(self, data):
        n = self.n
        data['input'] = cv2.resize(data['input'], (n, n), interpolation=cv2.INTER_LINEAR)
        data['mask'] = cv2.resize(data['mask'], (n, n), interpolation=cv2.INTER_NEAREST)
        return data


def compute_padding(h, w, n=128):
    if h % n == 0:
        dy0, dy1 = 0, 0
    else:
        dy = n - h % n
        dy0 = dy // 2
        dy1 = dy - dy0

    if w % n == 0:
        dx0, dx1 = 0, 0
    else:
        dx = n - w % n
        dx0 = dx // 2
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1


class PadToNxN(object):
    """Apply Pad to image size NxN using border reflection.
    Note:
        uses copyMakeBorder which and BORDER_REFLECT_101 which basically reflects the border of the image to pad.
    """

    def __init__(self, n=128):
        self.n = n

    def __call__(self, data):
        n = self.n
        h, w = data['input'].shape[:2]
        dy0, dy1, dx0, dx1 = compute_padding(h, w, n)

        data['input'] = cv2.copyMakeBorder(data['input'], dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
        data['mask'] = cv2.copyMakeBorder(data['mask'], dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
        return data


class HorizontalFlip(object):
    """Flip input and masks horizontally."""

    def __call__(self, data):
        data['input'] = cv2.flip(data['input'], 1)
        data['mask'] = cv2.flip(data['mask'], 1)
        return data


class BrightnessShift(object):
    """Applies Brightness shift to the images.
    Note:
        When changing the brightness of an image, a constant is added or subtracted from the luminnance of all
        sample values. Here we are shifting the histogram left (subtraction) or right (addition) by a max value.
    """

    def __init__(self, max_value=0.1):
        self.max_value = max_value

    def __call__(self, data):
        img = data['input']
        img += np.random.uniform(-self.max_value, self.max_value)
        data['input'] = np.clip(img, 0, 1)
        return data


class BrightnessScaling(object):
    """Applies Brightness scaling to the images.
    Note:
        Brightness scaling scales the histogram by a max value.
    """

    def __init__(self, max_value=0.08):
        self.max_value = max_value

    def __call__(self, data):
        img = data['input']
        img *= np.random.uniform(1 - self.max_value, 1 + self.max_value)
        data['input'] = np.clip(img, 0, 1)
        return data


class GammaChange(object):
    """Applies Gamma change to the images.
    Note:
        is a nonlinear operation used to encode and decode luminance values in images.
    """

    def __init__(self, max_value=0.08):
        self.max_value = max_value

    def __call__(self, data):
        img = data['input']
        img = img ** (1.0 / np.random.uniform(1 - self.max_value, 1 + self.max_value))
        data['input'] = np.clip(img, 0, 1)
        return data


def do_elastic_transform(image, mask, grid=10, distort=0.2):
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width, np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * (1 + random.uniform(-distort, distort))

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = int(grid)
    yy = np.zeros(height, np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * (1 + random.uniform(-distort, distort))

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    # grid
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                      borderValue=(0, 0, 0,))
    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
                     borderValue=(0, 0, 0,))

    # mask = (mask > 0.5).astype(np.float32)
    return image, mask


class ElasticDeformation(object):
    """Applies Elastic deformation to the images.
    Note:
        Elastic deformation of images as described in [Simard2003]_ (with modifications).
        Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    def __init__(self, grid=10, max_distort=0.15):
        self.grid = grid
        self.max_distort = max_distort

    def __call__(self, data):
        distort = np.random.uniform(0, self.max_distort)
        img, mask = do_elastic_transform(data['input'], data['mask'], self.grid, distort)

        data['input'] = img
        data['mask'] = mask
        return data


def do_rotation_transform(image, mask, angle=0):
    height, width = image.shape[:2]
    cc = np.cos(angle / 180 * np.pi)
    ss = np.sin(angle / 180 * np.pi)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2, height / 2])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101,
                                borderValue=(0, 0, 0,))
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101,
                               borderValue=(0, 0, 0,))
    # mask = (mask > 0.5).astype(np.float32)
    return image, mask


class Rotation(object):
    """Applies to the Rotation to the images.
    Note:
        Does rotation transformation.
    """

    def __init__(self, max_angle=15):
        self.max_angle = max_angle

    def __call__(self, data):
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        img, mask = do_rotation_transform(data['input'], data['mask'], angle)

        data['input'] = img
        data['mask'] = mask
        return data


def do_horizontal_shear(image, mask, scale=0):
    height, width = image.shape[:2]
    dx = int(scale * width)

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = np.array([[+dx, 0], [width + dx, 0], [width - dx, height], [-dx, height], ], np.float32)

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101, borderValue=(0, 0, 0,))
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101, borderValue=(0, 0, 0,))
    # mask = (mask > 0.5).astype(np.float32)
    return image, mask


class HorizontalShear(object):
    """Applies Horizontal Shear to the images.
    Note:
        horizontal shear (or shear parallel to the x axis) is a function that takes a generic point with coordinates
        (x,y) to the point (x+my,y); where m is a fixed parameter, called the shear factor.
    """
    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        scale = np.random.uniform(-self.max_scale, self.max_scale)
        img, mask = do_horizontal_shear(data['input'], data['mask'], scale)

        data['input'] = img
        data['mask'] = mask
        return data


class HWCtoCHW(object):
    """Converts HWC to CHW."""
    def __call__(self, data):
        data['input'] = data['input'].transpose((2, 0, 1))
        return data


def augmentations(args):
    """Applies random augmentations for the input images based on the transform probability.
    Note:
        Many methods are taken from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63974.
        The user can specify between geometric, image or both types of transforms to the images since sometimes
        some transformations work well for certain datasets.
    :param args:
        image_size (int)        : size of the image to be resized.
        transform_prob (float)  : probability to apply transformations on the data.
    :return:
        a compose of transformations.
    """

    augment_type = 'geometric'
    transform_prob = args.transform_prob
    if augment_type == 'geometric':
        geometric_transforms = Compose([RandomApply([HorizontalShear(max_scale=0.07)], p=transform_prob),
                                        RandomApply([Rotation(max_angle=15)], p=transform_prob),
                                        RandomApply([ElasticDeformation(max_distort=0.15)], p=transform_prob),
                                        ResizeToNxN(args.image_size),
                                        ConvertToTensor()
                                        ])

        return geometric_transforms

    elif augment_type == 'image':
        brightness_transform = Compose([RandomApply([BrightnessShift(max_value=0.1)], p=transform_prob),
                                        RandomApply([BrightnessScaling(max_value=0.08)], p=transform_prob),
                                        RandomApply([GammaChange(max_value=0.08)], p=transform_prob),
                                        ResizeToNxN(args.image_size),
                                        ConvertToTensor()
                                        ])

        return brightness_transform

    elif augment_type == 'both':
        both_transforms = Compose([RandomApply([HorizontalShear(max_scale=0.07)], p=transform_prob),
                                   RandomApply([Rotation(max_angle=15)], p=transform_prob),
                                   RandomApply([ElasticDeformation(max_distort=0.15)], p=transform_prob),
                                   RandomApply([BrightnessShift(max_value=0.1)], p=transform_prob),
                                   RandomApply([BrightnessScaling(max_value=0.08)], p=transform_prob),
                                   RandomApply([GammaChange(max_value=0.08)], p=transform_prob),
                                   ResizeToNxN(args.image_size),
                                   ConvertToTensor()
                                   ])

        return both_transforms
