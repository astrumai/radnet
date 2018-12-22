""""
I need to write a class for filter visualizer. Robust to different depths and capable of outputting different types
of maps based on users choice.

The user will call a function outside of the class, like the aguments and specify the output image type and the depth
or specific depth he wants to visualize

Notes:
    Depending on teh depth I will have to resize the image strarting from lets' say we start at 64 > 128 > 256
    The downsample and upsample block has different types of blocks so there has to be two types of functions for those
    as well.
"""
from PIL import Image
import tifffile as tiff
import torch
from torchvision.transforms import *
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(args):
    checkpoint = os.path.join(args.weights_dir, "./u_net_model.pt")
    if os.path.isfile(checkpoint):
        print("===> loading model '{}' ".format(checkpoint))
        model = torch.load(checkpoint)
        model = model[0]['model']
    else:
        raise FileNotFoundError("===> no model found at {}. Check model path or start training".format(checkpoint))
    return model


# plot down blocks
def plot_down_blocks(image, input_size):
    _transform = Compose([Resize(input_size), ToTensor()])
    trans_img = _transform(image)
    sqz_img = trans_img.unsqueeze(0).to(device)
    return sqz_img


# a function to pick the blocks for interpreting
def blocks(sampling, depth, model, img_path, args):
    kit = Image.fromarray((tiff.imread(img_path)))
    layers = list(model.children())

    if sampling == 'down':
        # if downsampling: call a function that takes in the depth of the model and splits it into respective blocks
        layer = layers[0]
        assert depth <= args.depth, "Depth should be equal to or less than the u-net depth"
        for d in depth:
            temp = list(layer.children())[d]

        pass

    elif sampling == 'up':
        # if upsampling: do the same but will have to add another child into it since the blocks has upsampling in it
        # need to see how to figure that out
        layer = layers[1]
        assert depth <= (args.depth-1), "Up sampling depth should be one less than the u-net depth"

        pass

    elif sampling == 'both':
        # if both call both sequentially

        pass


def sensitivity_analysis(model, image_tensor, target_class=None, postprocess='abs'):
    """
    Note:
        Code is based on "https://github.com/jrieke/cnn-interpretability/blob/master/interpretation.py"
    :param model:
    :param image_tensor:
    :param target_class:
    :param postprocess:
    :return:
    """
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor
    X = Variable(image_tensor[None], requires_grad=True)  # add dimension to simulate batch
    model = model.cpu()
    model.eval()
    output = model(X)
    output_class = output.max(1)[1].data.numpy()[0]
    # print('Image was classified as:', output_class)

    model.zero_grad()
    one_hot_output = torch.zeros(output.size())
    if target_class is None:
        one_hot_output[0, output_class] = 1
    else:
        one_hot_output[0, target_class] = 1
    output.backward(gradient=one_hot_output)

    relevance_map = X.grad.data[0].numpy()

    if postprocess == 'abs':  # as in Simonyan et al. (2013)
        return np.abs(relevance_map)
    elif postprocess == 'square':  # as in Montavon et al. (2018)
        return relevance_map**2
    elif postprocess is None:
        return relevance_map
    else:
        raise ValueError()


def plot_sensitivity(train_path, model, args):
    img = Image.fromarray(tiff.imread(train_path))
    img_transform = Compose([Resize(args.image_size), ToTensor()])
    img_tensor = img_transform(img)

    _mapped = sensitivity_analysis(model, img_tensor)
    mapped = np.squeeze(_mapped, axis=0)
    plt.imshow(mapped, cmap='gist_heat')
    plt.axis('off')
    plt.show()


# main function the caller will use
def interpret(train_path, args):

    model = load_model(args)

    if args.plot_interpret == 'sensitivity':
        plot_sensitivity(train_path, model, args)
