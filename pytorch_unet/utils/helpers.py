import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor


def pred_to_numpy(prediction):
    """
    :param prediction:
    :return:
    """
    return prediction.sigmoid().detach().cpu().numpy()


def to_numpy(tensor):
    """
    :param tensor:
    :return:
    """
    return tensor.detach().cpu().numpy()


def to_tuple(param, low=None):
    """
    :param param:
    :param low:
    :return:
    """

    if isinstance(param, (list, tuple)):
        return tuple(param)
    elif param is not None:
        if low is None:
            return -param, param
        return (low, param) if low < param else (param, low)
    else:
        return param


def convert_2d_to_3d(arrays, num_channels=3):
    """ Converts a 2D numpy array with shape (H, W) into a 3D array with shape (H, W, num_channels)
    by repeating the existing values along the new axis.

    :param arrays:
    :param num_channels:
    :return:
    """

    arrays = tuple(np.repeat(array[:, :, np.newaxis], repeats=num_channels, axis=2) for array in arrays)
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def convert_2d_to_target(arrays, target):
    """
    :param arrays:
    :param target:
    :return:
    """
    if target == 'mask':
        return arrays[0] if len(arrays) == 1 else arrays
    elif target == 'image':
        return convert_2d_to_3d(arrays, num_channels=3)
    elif target == 'image_4_channels':
        return convert_2d_to_3d(arrays, num_channels=4)
    else:
        raise ValueError('Unknown target {}'.format(target))


def plot_output(prediction):
    plt.axis('off')
    plt.imshow(prediction[0][0].cpu().detach().numpy())
    plt.show()


def save_model(model, path, epoch, optimizer, best, loss):
    if best:
        print("===> Saving a new best model at epoch {}".format(epoch))
        save_checkpoint = ({'model': model,
                            'optimizer': optimizer,
                            'epoch': epoch,
                            'best_loss': loss
                            }, best)
        torch.save(save_checkpoint, path + "/unet_model.pt")


def load_model(args):
    checkpoint = os.path.join(args.weights_dir, "./unet_model.pt")
    if os.path.isfile(checkpoint):
        print("===> loading model '{}' ".format(checkpoint))
        model = torch.load(checkpoint)
        model = model[0]['model']
    else:
        raise FileNotFoundError("===> no model found at {}. Check model path or start training".format(checkpoint))
    return model


def resume_training(args):
    """add in future"""
    if args.resume == 'yes':
        filename = os.path.join(args.weights_dir, "./unet_model.pt")
        if os.path.isfile(filename):
            print("===> loading checkpoint '{}' ".format(filename))
            checkpoint = torch.load(filename)
            start_epoch = checkpoint[0]['epoch']
            model = checkpoint[0]['model']
            optimizer = checkpoint[0]['optimizer']
            loss = checkpoint[0]['loss']
            print("===> loaded checkpoint '{}' (epoch {})".format(filename, start_epoch))
        else:
            raise FileNotFoundError("===> no model found at {}. Check model path or start training".format(filename))
        return model, optimizer, loss, start_epoch


def load_image(img_path, args):
    img = Image.fromarray((tiff.imread(img_path))[1])
    img_transform = Compose([Resize(args.image_size), ToTensor()])
    img_tensor = img_transform(img)
    return img_tensor, img
