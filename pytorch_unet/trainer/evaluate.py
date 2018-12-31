import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "pytorch_unet.trainer"

from pytorch_unet.processing.load import DataTransformer
from pytorch_unet.utils.helpers import pred_to_numpy, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for evaluating the trained model')

    parser.add_argument('--main_dir', default="C:\\Users\\Mukesh\\Segmentation\\UNet\\", help='main directory')
    parser.add_argument('--image_size', default=64, type=int, help='resize image size to match train image size')
    parser.add_argument('--weights_dir', default="./weights", type=str, help='Choose directory to save weights model')

    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    test_path = os.path.join(args.main_dir, 'data', 'test-volume.tif')

    # compose the transforms for the train set and transform to a 4D tensor
    test_data = Compose([Resize(args.image_size), ToTensor()])
    test_transform = DataTransformer(test_path, labels_filename=None, image_transform=test_data,
                                     image_augmentation=None)
    test_loader = DataLoader(test_transform)

    # load the saved model and start making predictions, if model is not present call training
    model = load_model(args)

    # you need to use eval, it doesn't change the output in this case cz i dont have dropout or batchnorm and if i did
    # model.eval would have taken care of it
    model = model.eval()
    pred_list = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # load the model for predictions
            prediction = model(data)
            # applies the sigmoid function to the output
            labels = pred_to_numpy(prediction)
            re_labels = labels.reshape((64, 64))
            pred_list.append(re_labels)

    # save the predictions
    print("Evaluations complete")


if __name__ == '__main__':
    main()
