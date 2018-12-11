import numpy as np


def dice(image1, image2, empty_score=1.0):
    """

    :param image1:
    :param image2:
    :param empty_score:
    :return:
    """
    image1 = np.asarray(image1).astype(np.bool)
    image2 = np.asarray(image2).astype(np.bool)

    if image1.shape != image2.shape:
        raise ValueError("Shape Mismatch: image1 and image2 must have the same shape")

    union = image1.sum() + image2.sum()
    if union == 0:
        return empty_score

    intersection = np.logical_and(image1, image2)
    return 2.*intersection.sum() / union
