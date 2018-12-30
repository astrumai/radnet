import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from ..utils.helpers import load_model, load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def all_children(mod):
    """Return a list of all child modules of the model, and their children, and their children's children, ..."""
    children = []
    for idx, child in enumerate(mod.named_modules()):
        children.append(child)
    return children


def get_values(iterables, key_to_find):
    return list(filter(lambda z: key_to_find in z, iterables))


def get_block_list(child_list, args, sampling):
    block_list = []
    if sampling == 'down':
        down_seq = []
        for y in range(args.depth):
            down_seq.append(get_values(child_list, '{}_path.{}'.format(sampling, y)))

        for x in range(args.depth):
            block_list.append(down_seq[x][0][1])

    elif sampling == 'up':
        up_seq = []
        for y in range(args.depth - 1):
            up_seq.append(get_values(child_list, '{}_path.{}'.format(sampling, y)))

        for x in range(args.depth - 1):
            block_list.append(up_seq[x][0][1])

    return block_list


def plot_block(args, block, img_size, name):
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (50, 50)

    for i in range(img_size):
        fig.add_subplot(round(sqrt(img_size))+1, round(sqrt(img_size))+1, i + 1)
        plt.imshow(block[0][i].cpu().detach().numpy())
        plt.axis('off')
    fig.suptitle('{0} Block Filter Size {1}x{1}'.format(name, img_size), fontsize=25)
    plt.savefig(args.interpret_path + '/{}_block_filter_{}.png'.format(name, img_size))


def block_filters(model, img_path, args):
    module_list = all_children(model)
    img_tensor, _ = load_image(img_path, args)
    input_img = img_tensor.unsqueeze(0).to(device)

    down_list = get_block_list(module_list, args, sampling='down')
    up_list = get_block_list(module_list, args, sampling='up')

    dep_list = [down_list[0](input_img)]
    for i in range(1, len(down_list)):
        layer = down_list[i](dep_list[i-1])
        dep_list.append(layer)

    for j, m in zip(range(len(dep_list)), [64, 128, 256]):
        plot_block(args, dep_list[j], m, name='down')


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
        return relevance_map ** 2
    elif postprocess is None:
        return relevance_map
    else:
        raise ValueError()


def plot_sensitivity(img_path, model, args):
    img_tensor, img = load_image(img_path, args)

    _mapped = sensitivity_analysis(model, img_tensor)
    mapped = np.squeeze(_mapped, axis=0)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Input Image')
    ax1.axis('off')
    ax1.imshow(img)

    ax2.set_title('Sensitivity Map')
    ax2.axis('off')
    ax2.imshow(mapped, cmap='gist_gray')

    plt.savefig(args.interpret_path + '/sensitivity.png')


def interpret(args):
    model = load_model(args)
    img_path = os.path.join(args.root_dir, 'data', 'test-volume.tif')

    if args.plot_interpret == 'sensitivity':
        plot_sensitivity(img_path, model, args)

    if args.plot_interpret == 'block_filters':
        block_filters(model, img_path, args)
