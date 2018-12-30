import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from pytorch_unet.processing import DataTransformer
from pytorch_unet.utils import pred_to_numpy, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(args):
    test_path = os.path.join(args.root_dir, 'data', 'test-volume.tif')

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
    print("predictions complete")
