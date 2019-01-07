import numpy as np


def dice(image1, image2, empty_score=1.0):
    """Dice score implementation.
    Note:
        The F1 score is also known as the Sørensen–Dice coefficient or Dice similarity coefficient (DSC).
    :param image1           : The prediction
    :param image2           : The label
    :param empty_score      : is to prevent division by 0

    :return                 : Dice scores
    """
    image1 = np.asarray(image1).astype(np.bool)
    image2 = np.asarray(image2).astype(np.bool)

    if image1.shape != image2.shape:
        raise ValueError("Shape Mismatch: image1 and image2 must have the same shape")

    union = image1.sum() + image2.sum()
    if union == 0:
        return empty_score

    intersection = np.logical_and(image1, image2)
    return 2. * intersection.sum() / union
