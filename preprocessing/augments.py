from torchvision.transforms import *
import numpy as np
import torch
import cv2


def converter(image, mask):
    """converts the torch tensor to a pil image and then an array"""

    pil_image = ToPILImage()(image)
    arr_image = np.asarray(pil_image)

    pil_mask = ToPILImage()(mask)
    arr_mask = np.asarray(pil_mask)

    return arr_image, arr_mask


def deconverter(image, mask):
    """converts the array to torch tensor"""

    tensor_image = torch.from_numpy(image)
    tensor_image.unsqueeze_(0)
    tensor_image = tensor_image.expand(1, 64, 64)
    tensor_mask = torch.from_numpy(mask)
    tensor_mask.unsqueeze_(0)
    tensor_mask = tensor_mask.expand(1, 64, 64)

    return tensor_image, tensor_mask


def normalize(image):
    """Need to normalize and round the output to 4 decimal points to match with validation data set"""
    norm_image = cv2.normalize(image, image.shape, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    round_num = np.around(norm_image, 4)
    return round_num


def rotation_transforms(image, mask, angle=0):
    height, width = image.shape[1:]
    cc = np.cos(angle / 180 * np.pi)
    ss = np.sin(angle / 180 * np.pi)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2, height / 2])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    # calculates a perspective transform from four pairs of the corresponding points
    mat = cv2.getPerspectiveTransform(box0, box1)

    # calling the converter
    arr_image, arr_mask = converter(image, mask)

    # applies the perspective transformation to the image and the mask
    image = cv2.warpPerspective(arr_image, mat, (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101,
                                borderValue=(0, 0, 0,))
    # normalizing the images
    image = normalize(image)

    mask = cv2.warpPerspective(arr_mask, mat, (width, height),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101,
                               borderValue=(0, 0, 0,))
    # normalizing the masks
    mask = normalize(mask)

    # convert the nd array to tensor
    tensor_image, tensor_mask = deconverter(image, mask)
    return tensor_image, tensor_mask


class Rotation(object):
    """ Do rotations

    """

    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, data):
        angle = np.random.uniform(-self.max_angle, self.max_angle)

        for i in range(len(data)):
            img, mask = rotation_transforms(data[i][0], data[i][1], angle)
            data[i][0] = img
            data[i][1] = mask
        return data


def elastic_transform(image, mask, grid=10, distort=0.2):
    height, width = image.shape[1:]
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
            cur = prev + x_step * (1 + np.random.uniform(-distort, distort))
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
            cur = prev + y_step * (1 + np.random.uniform(-distort, distort))

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur
    # grid
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # calling the converter
    arr_image, arr_mask = converter(image, mask)

    image = cv2.remap(arr_image, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101,
                      borderValue=(0, 0, 0,))
    # normalizing the images
    image = normalize(image)

    mask = cv2.remap(arr_mask, map_x, map_y,
                     interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_REFLECT_101,
                     borderValue=(0, 0, 0,))
    # normalizing the masks
    mask = normalize(mask)

    # convert the nd array to tensor
    tensor_image, tensor_mask = deconverter(image, mask)

    return tensor_image, tensor_mask


class ElasticDeformation(object):
    """ Do Elastic Transformations

    """

    def __init__(self, grid=10, max_distort=0.15):
        self.grid = grid
        self.max_distort = max_distort

    def __call__(self, data):
        distort = np.random.uniform(0, self.max_distort)
        for i in range(len(data)):
            img, mask = elastic_transform(data[i][0], data[i][1], self.grid, distort)
            data[i][0] = img
            data[i][1] = mask
        return data


class BrightnessShift(object):
    """

    """

    def __init__(self, max_value=0.1):
        self.max_value = max_value

    def __call__(self, data):
        for i in range(len(data)):
            img = data[i][0]
            img += np.random.uniform(-self.max_value, self.max_value)
            data[i][0] = np.clip(img, 0, 1)
        return data


class BrightnessScaling(object):
    """

    """

    def __init__(self, max_value=0.08):
        self.max_value = max_value

    def __call__(self, data):
        for i in range(len(data)):
            img = data[i][0]
            img *= np.random.uniform(1 - self.max_value, 1 + self.max_value)
            data[i][0] = np.clip(img, 0, 1)
        return data


class GammaChange(object):
    """

    """

    def __init__(self, max_value=0.08):
        self.max_value = max_value

    def __call__(self, data):
        for i in range(len(data)):
            img = data[i][0]
            img = img ** (1.0 / np.random.uniform(1 - self.max_value, 1 + self.max_value))
            data[i][0] = np.clip(img, 0, 1)
        return data


def augmentations(train):
    """Augmentations for the input images
    Note:


    Arguments:


    Returns:
    """
    augment_type = 'geometric'

    if augment_type == 'geometric':
        geometric_transforms = Compose([Rotation(max_angle=15),
                                        ElasticDeformation(max_distort=0.15)])

        return geometric_transforms(train)

    elif augment_type == 'brightness':
        bright_transforms = Compose([BrightnessShift(max_value=0.1),
                                    BrightnessScaling(max_value=0.08),
                                    GammaChange(max_value=0.08)])

        return bright_transforms(train)

    elif augment_type == 'both':
        both_transforms = Compose([Rotation(max_angle=15),
                                   ElasticDeformation(max_distort=0.15),
                                   BrightnessShift(max_value=0.1),
                                   BrightnessScaling(max_value=0.08),
                                   GammaChange(max_value=0.08)])

        return both_transforms(train)
