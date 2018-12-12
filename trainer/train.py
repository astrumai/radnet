from processing.load import DataTransformer
from model.u_net import UNet
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import os
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.optim import Adam
from processing.augments import augmentations
from sklearn.model_selection import train_test_split
from utils.helpers import pred_to_numpy, to_numpy
from utils.metrics import dice
import numpy as np
from utils.logger import my_logs

# CUDA for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create the loss function
loss_bce = nn.BCEWithLogitsLoss()
loss_values = []
THRESHOLD = 0.5
best_loss = 1e10


def save_models(model, path, epoch, optimizer, best, loss):
    if best:
        print("===> Saving a new best model at epoch {}".format(epoch))
        save_checkpoint = ({'model': model,
                            'optimizer': optimizer,
                            'epoch': epoch,
                            'best_loss': loss
                            }, best)
        torch.save(save_checkpoint, path + "/u_net_model.pt")


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


def train(args):
    """ Contains the training loop"""
    global best_loss

    # get data set file path
    data_path = os.path.join(args.root_dir, 'data', 'train-volume.tif')
    labels_path = os.path.join(args.root_dir, 'data', 'train-labels.tif')

    # compose the transforms for the train set
    train_data = Compose([Resize(args.image_size), ToTensor()])

    # choose between augmentations for train data
    if args.augment == 'yes':
        train_augment = augmentations()
        train_transform = DataTransformer(data_path, labels_path, image_transform=train_data,
                                          image_augmentation=train_augment)

    elif args.augment == 'no':
        # transforming the train data and returning a 4D tensor
        train_transform = DataTransformer(data_path, labels_path, image_transform=train_data, image_augmentation=None)

    # transform for validation data
    val_data = Compose([Resize(args.image_size), ToTensor()])
    val_transform = DataTransformer(data_path, labels_path, image_transform=val_data, image_augmentation=None)

    # split the train and validation indices
    train_indices, validation_indices = train_test_split(range(len(train_transform)), test_size=0.15)

    # call the sampler for the train and validation data
    train_samples = RandomSampler(train_indices)
    validation_samples = SequentialSampler(validation_indices)

    # load train and validation data
    train_loader = DataLoader(train_transform, batch_size=args.batch_size, sampler=train_samples)
    val_loader = DataLoader(val_transform, batch_size=args.batch_size, sampler=validation_samples)

    # create the model
    model = UNet(in_channels=1, n_classes=args.n_classes, depth=args.depth, wf=6, padding=True, batch_norm=False,
                 up_mode=args.up_mode).to(device)

    # create the optimizer
    optim = Adam(model.parameters())

    # start the training loop
    for e in range(args.epochs):
        for i, data in enumerate(train_loader):
            train_batch, train_labels = data
            train_batch, train_labels = train_batch.to(device), train_labels.to(device)

            # call the u-net module
            prediction = model(train_batch)

            # call the loss function
            loss = loss_bce(prediction, train_labels)
            loss_values.append(loss)

            # backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

        # print the loss
        train_dice, train_loss = validate_model(model, train_loader, threshold=THRESHOLD)
        val_dice, val_loss = validate_model(model, val_loader, threshold=THRESHOLD)
        print("===> Epoch {} Training Loss: {:.4f} : Mean Dice: {}".format(e, train_loss, train_dice))
        print("===> Epoch {} Validation Loss: {:.4f} : Mean Dice: {} :".format(e, val_loss, val_dice))

        # save model with best score
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_models(model=model, path=args.weights_dir, epoch=e + 1, optimizer=optim, best=is_best, loss=best_loss)

        # log the values
        history = my_logs(epoch=e, train_loss=train_loss, train_dice=train_dice, val_loss=val_loss, val_dice=val_dice)

    print("Training values logged to train.")

    # plot train values
    if args.plotter == 'yes':
        from utils.plot import plotter

        show_plots = plotter(history)
    elif args.plotter == 'no':
        print("Nothing plotted")
