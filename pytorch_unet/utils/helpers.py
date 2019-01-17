import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor


def pred_to_numpy(prediction):
    """Converts the predictions to values between [0, 1]."""
    return prediction.sigmoid().detach().cpu().numpy()


def to_numpy(tensor):
    """Converts the tensor to numpy."""
    return tensor.detach().cpu().numpy()


def to_tuple(param, low=None):
    """Converts a parameter to a tuple."""

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
    by repeating the existing values along the new axis."""

    arrays = tuple(np.repeat(array[:, :, np.newaxis], repeats=num_channels, axis=2) for array in arrays)
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def convert_2d_to_target(arrays, target):
    """Converts a 2D array to any specified target array of target shape."""
    if target == 'mask':
        return arrays[0] if len(arrays) == 1 else arrays
    elif target == 'image':
        return convert_2d_to_3d(arrays, num_channels=3)
    elif target == 'image_4_channels':
        return convert_2d_to_3d(arrays, num_channels=4)
    else:
        raise ValueError('Unknown target {}'.format(target))


def plot_output(prediction):
    """Plots the output for the prediction."""
    plt.axis('off')
    plt.imshow(prediction[0][0].cpu().detach().numpy())
    plt.show()


def save_model(model, path, epoch, optimizer, best, loss, verbose):
    """Saves the model, optimizer, epochs and the best loss."""
    if best:
        save_checkpoint = ({'model': model,
                            'optimizer': optimizer,
                            'epoch': epoch,
                            'best_loss': loss
                            }, best)
        torch.save(save_checkpoint, path + "/unet_model.pt")

        if verbose:
            print("*** Saving a new best model at epoch {} ***\n".format(epoch))


def load_model(args):
    """Loads the saved model."""
    checkpoint = os.path.join(args.weights_dir, "./unet_model.pt")
    if os.path.isfile(checkpoint):
        print("===> loading model '{}' ".format(checkpoint))
        model = torch.load(checkpoint)
        model = model[0]['model']
    else:
        raise FileNotFoundError("===> no model found at {}. Check model path or start training".format(checkpoint))
    return model


def resume_training(args):
    """Function to be added in for future."""
    if args.resume:
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
    """Loads the image from the image path.
    Note:
        this function is used only in the interpret.py and a point to note is here you are specifying the img since
        in this dataset the data is in the volume format so you need the indexing to select the data but for another
        dataset you need to change this.
    """
    img = Image.fromarray((tiff.imread(img_path))[1])
    img_transform = Compose([Resize(args.image_size), ToTensor()])
    img_tensor = img_transform(img)
    return img_tensor, img


def timeit_decor(function, *args, **kwargs):
    """Decorator function to print exectution time of the decorated function.
    Usage:
        @timeit_decor on top of the function
    
    :param func: [description]
    :return: [description]
    """

    def timer(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        t2 = time.time()
        print("-- executed %s in %.4f seconds" % (function.__name__, (t2 - t1)))
        return result

    return timer


def elapsed_since(start):
    """Helper to calculate the elapsed time in ms, s, min or hrs.
    
    :param start (sec)      : starting time
    :return                 : elapsed time in ms, s, min or hrs
    """

    elapsed = time.time() - start
    if elapsed < 1:
        return str(round(elapsed * 1000, 2)) + 'ms'
    if elapsed < 60:
        return str(round(elapsed, 2)) + 's'
    if elapsed < 3600:
        return str(round(elapsed / 60, 2)) + 'min'
    else:
        return str(round(elapsed / 3600, 2)) + 'hrs'


def format_bytes(bytes):
    """Helper to convert bytes into respective formats.
    
    :param bytes(byte)  : Bytes to formate
    :return (byte)      : formatted bytes
    """

    if abs(bytes) < 1000:
        return str(bytes) + 'B'
    elif abs(bytes) < 1e6:
        return str(round(bytes / 1e3, 2)) + 'kB'
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + 'MB'
    else:
        return str(round(bytes / 1e9, 2)) + 'GB'


def _find_script(script_name):
    """ Find the script.
    If the input is not a file, then $PATH will be searched.
    """
    if os.path.isfile(script_name):
        return script_name
    path = os.getenv('PATH', os.defpath).split(os.pathsep)
    for folder in path:
        if not folder:
            continue
        fn = os.path.join(folder, script_name)
        if os.path.isfile(fn):
            return fn

    sys.stderr.write('Could not find script {0}\n'.format(script_name))
    raise SystemExit(1)

# import pickle, gc, os, signal, threading, time, tracemalloc

# class TakeSnapshot(threading.Thread):
#     daemon = True

#     def run(self):
#         if hasattr(signal, 'pthread_sigmask'):
#             # Available on UNIX with Python 3.3+
#             signal.pthread_sigmask(signal.SIG_BLOCK, range(1, signal.NSIG))
#         counter = 1
#         while True:
#             time.sleep(60)
#             filename = ("/tmp/tracemalloc-%d-%04d.pickle"
#                         % (os.getpid(), counter))
#             print("Write snapshot into %s..." % filename)
#             gc.collect()
#             snapshot = tracemalloc.take_snapshot()
#             with open(filename, "wb") as fp:
#                 pickle.dump(snapshot, fp, 2)
#             snapshot = None
#             print("Snapshot written into %s" % filename)
#             counter += 1

# save 25 frames
# tracemalloc.start(25)
# TakeSnapshot().start()
