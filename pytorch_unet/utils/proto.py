from math import sqrt

import matplotlib.pyplot as plt
import tifffile as tiff
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision.transforms import Resize, ToTensor, Compose

from pytorch_unet.model.u_net import UNet
from pytorch_unet.processing.load import DataTransformer

torch.set_default_tensor_type('torch.FloatTensor')

data_path = "C:\\Users\\Mukesh\\Segmentation\\radnet\\data\\"

trainPath = data_path + 'train-volume.tif'
labelsPath = data_path + 'train-labels.tif'

# print("shape:", (tiff.imread(trainPath)).shape)
# print("Pil image 1: ", Image.fromarray((tiff.imread(trainPath))[1]))
# print("Pil image 2: ", Image.fromarray((tiff.imread(trainPath))[2]))
# print("Pil image 29: ", Image.fromarray((tiff.imread(trainPath))[29]))

train_data = Compose([Resize(64), ToTensor()])

# choose between augmentations for train data
# train_augment = augmentations()

train_transform = DataTransformer(trainPath, labelsPath, image_transform=train_data,
                                  image_augmentation=None)

# split the train and validation indices
train_indices, validation_indices = train_test_split(range(len(train_transform)), test_size=0.15)

# call the sampler for the train and validation data
train_samples = RandomSampler(train_indices)

# load train and validation data
train_loader = DataLoader(train_transform,
                          batch_size=1,
                          sampler=train_samples)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create the model
model = UNet(in_channels=1,
             n_classes=1,
             depth=3,
             wf=6,
             padding=True,
             batch_norm=False,
             up_mode='upsample').to(device)

# create the optimizer
optim = Adam(model.parameters())
loss_bce = nn.BCEWithLogitsLoss()
loss_values = []
epochs = 1
# start the training loop
for e in range(epochs):
    epoch_loss = 0.0

    for i, data in enumerate(train_loader):
        train_batch, train_labels = data
        train_batch, train_labels = train_batch.to(device), train_labels.to(device)

        # call the u-net module
        prediction = model(train_batch)

        # zero the parameter gradients
        optim.zero_grad()

        # call the loss function
        loss = loss_bce(prediction, train_labels)
        loss_values.append(loss)

        # backprop the loss
        loss.backward()
        optim.step()


def all_children(mod):
    """Return a list of all child modules of the model, and their children, and their children's children, ..."""
    children = []
    for idx, child in enumerate(mod.named_modules()):
        children.append(child)
    return children


def get_values(iterables, key_to_find):
    return list(filter(lambda z: key_to_find in z, iterables))


def get_block_list(child_list, depth):
    down_block_list = []
    up_block_list = []
    down_seq = []
    for y in range(depth):
        down_seq.append(get_values(child_list, 'down_path.{}'.format(y)))

    for x in range(depth):
        down_block_list.append(down_seq[x][0][1])

    up_seq = []
    for y in range(depth - 1):
        up_seq.append(get_values(child_list, 'up_path.{}'.format(y)))

    for x in range(depth - 1):
        up_block_list.append(up_seq[x][0][1])

    return down_block_list, up_block_list


def plot_block(args, block, img_size, name):
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (50, 50)

    for i in range(img_size):
        fig.add_subplot(round(sqrt(img_size)) + 1, round(sqrt(img_size)) + 1, i + 1)
        plt.imshow(block[0][i].cpu().detach().numpy())
        plt.axis('off')
    fig.suptitle('{0} Block Filter Size {1}x{1}'.format(name, img_size), fontsize=25)
    plt.savefig(args.interpret_path + '/{}_block_filter_{}.png'.format(name, img_size))


module_list = all_children(model)
# che = model.named_parameters()
# for k,v in iter(che):
#     print(k)

img = Image.fromarray((tiff.imread(trainPath))[1])
img_transform = Compose([Resize(64), ToTensor()])
img_tensor = img_transform(img)
input_img = img_tensor.unsqueeze(0).to(device)

down_list, up_list = get_block_list(module_list, 3)
# print(up_list[0])
# print(down_list[2])
# print(down_list[1])


dep_list = [down_list[0](input_img)]
for i in range(1, len(down_list)):
    layer = down_list[i](dep_list[i - 1])
    dep_list.append(layer)

# apply pooling to everything inside dep list
pooled = []
for i in range(0, len(dep_list)):
    x = F.avg_pool2d(dep_list[i], 2)
    pooled.append(x)

print("depth1: ", dep_list[1].shape)
x = F.avg_pool2d(dep_list[1], 2)

x2 = F.avg_pool2d(F.avg_pool2d(dep_list[2], 2), 2)
print("x2: ", x2.shape)
print("*****************")

print("depth2: ", dep_list[0].shape)
y = F.avg_pool2d(dep_list[0], 2)

y2 = F.avg_pool2d(F.avg_pool2d(dep_list[1], 2), 2)
print("y2: ", y2.shape)
print("*****************")

upth_list = up_list[0](F.avg_pool2d(F.avg_pool2d(dep_list[2], 2), 2), F.avg_pool2d(dep_list[1], 2))
upth_list2 = up_list[1](F.avg_pool2d(F.avg_pool2d(dep_list[1], 2), 2), F.avg_pool2d(dep_list[0], 2))


def pooling(dep, d):
    if (dep - 1) == 1:
        return F.avg_pool2d(d, 2)
    else:
        return F.avg_pool2d(pooling(dep - 1, d), 2)


upton = []
rev_uplist = up_list[::-1]
for i in range(len(up_list)):
    pools = pooling(3, dep_list[i + 1])
    up_layer = rev_uplist[i](pools, F.avg_pool2d(dep_list[i], 2))
    upton.append(up_layer)


def plot_block(block, img_size, name):
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (50, 50)

    for i in range(img_size):
        fig.add_subplot(round(sqrt(img_size)) + 1, round(sqrt(img_size)) + 1, i + 1)
        plt.imshow(block[0][i].cpu().detach().numpy())
        plt.axis('off')
    plt.show()


for j, m in zip(range(len(up_list)), [16, 32]):
    plot_block(upton[j], m, name='up')

# fig = plt.figure()
# plt.rcParams["figure.figsize"] = (50, 50)
#
# for i in range(32):
#     fig.add_subplot(round(sqrt(32)) + 1, round(sqrt(32)) + 1, i + 1)
#     plt.imshow(upth_list2[0][i].cpu().detach().numpy())
#     plt.axis('off')
# plt.show()


# kit = Image.fromarray((tiff.imread(trainPath))[1])
# _data = Compose([Resize(64), ToTensor()])
# kit2 = _data(kit)
# kit3 = kit2.unsqueeze(0).to(device)
#
# dep_list = [down_list[0](kit3)]
# for i in range(1, len(down_list)):
#     layer = down_list[i](dep_list[i - 1])
#     dep_list.append(layer)
#
# up_seq = []
# for y in range(2):
#     up_seq.append(get_values(childlist, 'up_path.{}'.format(y)))
#
# up_list = []
# for x in range(2):
#     up_list.append(up_seq[x][0][1])
#
# # print(up_list[0])
#
# up_depth_list = [dep_list[2]]
#
# up_depth_list0 = dep_list[2]

# up_depth_list1 = up_list[0](up_depth_list0, dep_list[2])
# up_depth_list2 = up_list[1](up_depth_list1, dep_list[1])


# fig = plt.figure()
# plt.rcParams["figure.figsize"] = (50, 50)
# for i in range(256):
#     fig.add_subplot(round(sqrt(256))+1, round(sqrt(256))+1, i + 1)
#     plt.imshow(up_depth_list[0][0][i].cpu().detach().numpy())
#     plt.axis('off')
# plt.show()


# image_vis = tem(kit3)
# print(image_vis.shape)
#
# fig = plt.figure()
# plt.rcParams["figure.figsize"] = (128, 128)
# """The output of the 1st block has 64 filters so by changing from [0][0 to 63] you can visualize those filters"""
#
# for i in range(128):
#     fig.add_subplot(8, 8, i+1)
#     imgplot = plt.imshow(image_vis[0][i].cpu().detach().numpy())
#     plt.axis('off')
# plt.show()


# for x in range(2):
#     for y in range(2):
#         seq.append(get_values(childlist, 'up_path.{}.up.{}'.format(x, y)))
# for x in range(2):
#     for y in range(4):
#         seq.append(get_values(childlist, 'up_path.{}.conv_block.block.{}'.format(x, y)))

# print(len(down_seq))
# print((down_seq[0][0][1]))
# vgg = models.vgg16(pretrained=True)
# modulelist = list(vgg.features.modules())
# print(modulelist)


# module_list = dict(model.named_parameters())
# layers = list(model.children())
# l = layers[0]
# tem = list(l.children())[1]
# tem2 = list(tem.children())
# print(tem)
#
# for k, v in module_list.items():
#     print(k, ":")
#
# kit = Image.fromarray((tiff.imread(trainPath))[1])
# _data = Compose([Resize(64), ToTensor()])
# kit2 = _data(kit)
# kit3 = kit2.unsqueeze(0).to(device)
#
# image_vis = tem(kit3)
# print(image_vis.shape)
#
# fig = plt.figure()
# plt.rcParams["figure.figsize"] = (128, 128)
# """The output of the 1st block has 64 filters so by changing from [0][0 to 63] you can visualize those filters"""
#
# for i in range(128):
#     fig.add_subplot(8, 8, i+1)
#     imgplot = plt.imshow(image_vis[0][i].cpu().detach().numpy())
#     plt.axis('off')
# plt.show()
