from torchvision.transforms import *
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from functools import wraps
import torch
import cv2

MAX_VALUES_BY_DTYPE = {
    np.dtype('uint8'): 255,
    np.dtype('uint16'): 65535,
    np.dtype('uint32'): 4294967295,
    np.dtype('float32'): 1.0,
}


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def preserve_shape(func):
    """Preserve shape of the image"""

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


def clipped(func):
    """ """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


@preserve_shape
def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
    else:
        img = np.power(img, gamma)
    return img


@clipped
def brightness_contrast_adjust(img, alpha=1, beta=0):
    img = img.astype('float32') * alpha + beta * np.mean(img)
    return img


@preserve_shape
def grid_distortion(image, num_steps=10, xsteps=[], ysteps=[], interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101):
    """
    :param image:
    :param num_steps:
    :param xsteps:
    :param ysteps:
    :param interpolation:
    :param border_mode:
    :return:
    """
    height, width = image.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for i, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[i]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for i, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[i]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

    return image


@preserve_shape
def elastic_transform_fast(image, alpha, sigma, alpha_affine, interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_REFLECT_101, random_state=None):
    """
    :param image:
    :param alpha:
    :param sigma:
    :param alpha_affine:
    :param interpolation:
    :param border_mode:
    :param random_state:
    :return:
    """
    if random_state is None:
        random_state = np.random.RandomState(1234)

    height, width = image.shape[:2]

    # Random affine
    center_square = np.float32((height, width)) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, matrix, (width, height), flags=interpolation, borderMode=border_mode)

    dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)
    dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)

    return cv2.remap(image, mapx, mapy, interpolation, borderMode=border_mode)


class DualTransform(object):
    @property
    def targets(self):
        return {'image': self.apply, 'mask': self.apply_to_mask}

    def apply(self, img, **params):
        raise NotImplementedError

    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()})


class ImageOnlyTransform(object):
    @property
    def targets(self):
        return {'image': self.apply}

    def apply(self, img, **params):
        raise NotImplementedError


class GammaChange(ImageOnlyTransform):
    """
    Arguments:

    Returns:
    """

    def __init__(self, gamma_limit=(80, 120)):
        self.gamma_limit = gamma_limit

    def apply(self, img, gamma=1, **params):
        return gamma_transform(img, gamma=gamma)


class BrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image
    Arguments:

    Returns:
    """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def apply(self, img, alpha=1, beta=0, **params):
        return brightness_contrast_adjust(img, alpha, beta)


class GridDistortion(DualTransform):
    """
    Arguments:

    Returns:
    """

    def __init__(self, num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, stepsx=[], stepsy=[], interpolation=cv2.INTER_LINEAR,
              border_mode=cv2.BORDER_REFLECT_101, **params):
        return grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation, self.border_mode)


class ElasticTransform(DualTransform):
    """

    """

    def __init__(self, alpha, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101):
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
        return elastic_transform_fast(img, self.alpha, self.sigma, self.alpha_affine, interpolation,
                                      self.border_mode, np.random.RandomState(random_state))


class Rotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times
    Arguments:

    Returns:
    """

    def apply(self, img, factor=0, **params):
        factor = np.random.randint(0, 3)
        return np.ascontiguousarray(np.rot90(img, factor))


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis
    Arguments:

    Returns:

    """

    def apply(self, img, **params):
        return np.ascontiguousarray(img[:, ::-1, ...])


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis
    Arguments:


    Returns:

    """

    def apply(self, img, **params):
        return np.ascontiguousarray(img[::-1, ...])


def augmentations():
    """Augmentations for the input images
    Note:


    Arguments:


    Returns:
    """
    augment_type = 'geometric'
    transform_prob = 0.5
    if augment_type == 'geometric':
        geometric_transforms = Compose([VerticalFlip(),
                                        HorizontalFlip(),
                                        Rotate90(),
                                        RandomApply([ElasticTransform()], p=transform_prob),
                                        RandomApply([GridDistortion()], p=transform_prob)])

        return geometric_transforms

    elif augment_type == 'image':
        bright_transforms = Compose([RandomApply([BrightnessContrast()], p=transform_prob),
                                     RandomApply([GammaChange()], p=transform_prob)])

        return bright_transforms

    elif augment_type == 'both':
        both_transforms = Compose([VerticalFlip(),
                                   HorizontalFlip(),
                                   Rotate90(),
                                   RandomApply([ElasticTransform()], p=transform_prob),
                                   RandomApply([GridDistortion()], p=transform_prob),
                                   RandomApply([BrightnessContrast()], p=transform_prob),
                                   RandomApply([GammaChange()], p=transform_prob)])

        return both_transforms
