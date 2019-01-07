import torch
import torch.nn.functional as F
from torch import nn


class UNet(nn.Module):
    """U-Net implementation.
    Note:
        You need the forward function here since the ModuleList doesn't have a forward method
        and it is just a python list.
    Arguments:
        in_channels (int)   : number of input filter channels.
        n_classes (int)     : number of classes to predict.
        depth (int)         : depth of the network given in terms of blocks.
        wf (int)            : number of filters in the first layer.
        padding (bool)      : if True, apply padding such that the input shape is the same as the output.
        batch_norm (bool)   : Use BatchNorm after layers with an activation function.
        up_mode (str)       : one of 'upconv' or 'upsample'.
                                'upconv' will use transposed convolutions for learned upsampling.
                                'upsample' will use bilinear upsampling.
    Returns:
        The complete unet block.
    """

    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False, batch_norm=False, up_mode='upconv'):

        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()

        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    """Implementation of the down convolution of the U-Net convolution block.
    Note:
        Creates a down conv block with conv2d layers and relu layers using sequential.
    Arguments:
        in_size (int)       : input size of the 4D tensor with [N,C,H,W].
        out_size (int)      : output size of the 4D tensor with [N,C,H,W].
        padding (bool)      : if True, apply padding such that the input shape is the same as the output.
        batch_norm (bool)   : Use BatchNorm after layers with an activation function.
    Returns:
        A Unet conv block with relu function and batch norm if batch norm is set to True.
    """

    def __init__(self, in_size, out_size, padding, batch_norm):

        super(UNetConvBlock, self).__init__()
        block = [nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)),
                 nn.ReLU()]

        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    """Implementation of the Up convolutional U-net block.
    Note:
        Creates the upsampling block by either using transposed convolution or upsampling and the function center crop
        takes the output of the down block and center crops it and concatenates it to the respective up conv block.
    Arguments:
        in_size (int)       : input size of the 4D tensor with [N,C,H,W].
        out_size (int)      : output size of the 4D tensor with [N,C,H,W].
        up_mode (str)       : one of 'upconv' or 'upsample'.
                                'upconv' will use transposed convolutions for learned upsampling.
                                'upsample' will use bilinear upsampling.
        padding (bool)      : if True, apply padding such that the input shape is the same as the output.
        batch_norm (bool)   : Use BatchNorm after layers with an activation function.
    Returns:
        The Up sampling block.
    """

    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):

        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    @staticmethod
    def center_crop(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
