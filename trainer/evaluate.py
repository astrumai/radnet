import os
from torchvision.transforms import Compose, Resize, ToTensor
from processing.load import DataTransformer
from torch.utils.data import DataLoader
import torch
from utils.helpers import pred_to_numpy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


