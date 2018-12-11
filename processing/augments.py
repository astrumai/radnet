from torchvision.transforms import *
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from functools import wraps
from utils.helpers import to_tuple
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
def optical_distortion(img, k=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101):
    height, width = img.shape[:2]

    fx = width
    fy = height

    cx = width * 0.5 + dx
    cy = height * 0.5 + dy

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)
    img = cv2.remap(img, map1, map2, interpolation=interpolation, borderMode=border_mode)
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


class BasicTransform(object):
    def __init__(self, always_apply=False, p=0.5):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets = {}

    def __call__(self, **kwargs):
        if (np.random.random() < self.p) or self.always_apply:
            params = self.get_params()
            params = self.update_params(params, **kwargs)
            if self.targets_as_params:
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)
            res = {}
            for key, arg in kwargs.items():
                if arg is not None:
                    target_function = self._get_target_function(key)
                    target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                    res[key] = target_function(arg, **dict(params, **target_dependencies))
                else:
                    res[key] = None
            return res
        return kwargs

    def _get_target_function(self, key):
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, None)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        if hasattr(self, 'interpolation'):
            params['interpolation'] = self.interpolation
        params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
        return params

    @property
    def target_dependence(self):
        return {}

    def add_targets(self, additional_targets):
        """Add targets to transform them the same way as one of existing targets
        ex: {'target_image': 'image'}
        ex: {'obj1_mask': 'mask', 'obj2_mask': 'mask'}
        by the way you must have at least one object with key 'image'
        Args:
            additional_targets (dict): keys - new target name, values - old target name. ex: {'image2': 'image'}
        """
        self._additional_targets = additional_targets

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError


class DualTransform(BasicTransform):
    def get_params_dependent_on_targets(self, params):
        pass

    def apply(self, img, **params):
        pass

    @property
    def targets(self):
        return {'image': self.apply, 'mask': self.apply_to_mask}

    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()})


class ImageOnlyTransform(BasicTransform):
    def get_params_dependent_on_targets(self, params):
        pass

    def apply(self, img, **params):
        pass

    @property
    def targets(self):
        return {'image': self.apply}


class GammaChange(ImageOnlyTransform):
    """
    Arguments:

    Returns:
    """

    def __init__(self, gamma_limit=(80, 120)):
        super().__init__()
        self.gamma_limit = gamma_limit

    def apply(self, img, gamma=1, **params):
        return gamma_transform(img, gamma=gamma)


class BrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image
    Arguments:

    Returns:
    """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2):
        super().__init__()
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def apply(self, img, alpha=1, beta=0, **params):
        return brightness_contrast_adjust(img, alpha, beta)


class OpticalDistortion(DualTransform):
    """
    Arguments:

    Returns:
    """

    def __init__(self, distort_limit=0.05, shift_limit=0.05, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101):
        super().__init__()
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, k=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return optical_distortion(img, k, dx, dy, interpolation, self.border_mode)

    def get_params(self):
        return {'k': np.random.uniform(self.distort_limit[0], self.distort_limit[1]),
                'dx': round(np.random.uniform(self.shift_limit[0], self.shift_limit[1])),
                'dy': round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]))}


class GridDistortion(DualTransform):
    """
    Arguments:

    Returns:
    """

    def __init__(self, num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101):
        super().__init__()
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, stepsx=[], stepsy=[], interpolation=cv2.INTER_LINEAR,
              border_mode=cv2.BORDER_REFLECT_101, **params):
        return grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation, self.border_mode)

    def get_params(self):
        stepsx = [1 + np.random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in
                  range(self.num_steps + 1)]
        stepsy = [1 + np.random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in
                  range(self.num_steps + 1)]
        return {'stepsx': stepsx, 'stepsy': stepsy}


class ElasticTransform(DualTransform):
    """

    """

    def __init__(self, alpha, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101):
        super().__init__()
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
        return elastic_transform_fast(img, self.alpha, self.sigma, self.alpha_affine, interpolation,
                                      self.border_mode, np.random.RandomState(random_state))

    def get_params(self):
        return {'random_state': np.random.randint(0, 10000)}


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
                                        RandomApply([ElasticTransform(alpha=0.15)], p=transform_prob),
                                        RandomApply([GridDistortion()], p=transform_prob),
                                        RandomApply([OpticalDistortion(distort_limit=1, shift_limit=1)])
                                        ])

        return geometric_transforms

    elif augment_type == 'image':
        bright_transforms = Compose([RandomApply([BrightnessContrast()], p=transform_prob),
                                     RandomApply([GammaChange()], p=transform_prob)
                                     ])

        return bright_transforms

    elif augment_type == 'both':
        both_transforms = Compose([VerticalFlip(),
                                   HorizontalFlip(),
                                   Rotate90(),
                                   RandomApply([ElasticTransform(alpha=0.15)], p=transform_prob),
                                   RandomApply([GridDistortion()], p=transform_prob),
                                   RandomApply([OpticalDistortion(distort_limit=1, shift_limit=1)]),
                                   RandomApply([BrightnessContrast()], p=transform_prob),
                                   RandomApply([GammaChange()], p=transform_prob)
                                   ])

        return both_transforms
