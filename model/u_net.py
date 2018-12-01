import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    """ U-Net implementation
    Arguments:
        in_channels (int): number of input channels
        n_classes (int): number of output channels
        depth (int): depth of the network
        wf (int): number of filters in the first layer is 2**wf
        padding (bool): if True, apply padding such that the input shape
                        is the same as the output.
                        This may introduce artifacts
        batch_norm (bool): Use BatchNorm after layers with an
                           activation function
        up_mode (str): one of 'upconv' or 'upsample'.
                       'upconv' will use transposed convolutions for
                       learned upsampling.
                       'upsample' will use bilinear upsampling.

    Returns:


    """

    def __int__(self,
                in_channels=1,
                n_classes=2,
                depth=5,
                wf=6,
                padding=False,
                batch_norm=False,
                up_mode='upconv'):

        super().__init__(self)
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels,
                                                2**(wf+1),
                                                padding,
                                                batch_norm))
            prev_channels = 2**(wf+i)


class UNetConvBlock(nn.Module):
    """ Implementation of the U-Net convolution block
    Arguments:
        prev_channels (int):
        padding (int):
        batch_norm ():

    Return:


    """



