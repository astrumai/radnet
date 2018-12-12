import os
from torchvision.transforms import Compose, Resize, ToTensor
from processing.load import DataTransformer
from torch.utils.data import DataLoader
import torch
import numpy as np
from utils.metrics import dice
from utils.helpers import pred_to_numpy, to_numpy
from torch import nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_bce = nn.BCEWithLogitsLoss()


def validate_model(model, loader, threshold):
    """ Contains the validation loop
    :param model:
    :param loader:
    :param threshold:
    :return:
    """
    model = model.eval()
    for i, data in enumerate(loader):
        val_batch, val_labels = data
        val_batch, val_labels = val_batch.to(device), val_labels.to(device)

        # call the u-net module
        prediction = model(val_batch)

        # call the loss function
    loss = loss_bce(prediction, val_labels)
    pred_labels, true_labels = pred_to_numpy(prediction) > threshold, to_numpy(val_labels) > threshold
    dices = []
    for _pred, _labels in zip(pred_labels, true_labels):
        dices.append(dice(_pred, _labels))
    return np.array(dices).mean(), loss.detach().cpu().numpy()


def evaluate(args):
    test_path = os.path.join(args.root_dir, 'data', 'test-volume.tif')

    # compose the transforms for the train set and transform to a 4D tensor
    test_data = Compose([Resize(args.image_size), ToTensor()])
    test_transform = DataTransformer(test_path, labels_filename=None, image_transform=test_data, image_augmentation=None)
    test_loader = DataLoader(test_transform)

    # load the saved model and start making predictions, if model is not present call training
    checkpoint = os.path.join(args.weights_dir, "./u_net_model.pt")
    if os.path.isfile(checkpoint):
        print("===> loading checkpoint '{}' ".format(checkpoint))
        model = torch.load(checkpoint)
        model = model[0]['model']
        model = model.eval()
    else:
        raise FileNotFoundError("===> no checkpoint found at {}".format(checkpoint))
    pred_list = []
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


