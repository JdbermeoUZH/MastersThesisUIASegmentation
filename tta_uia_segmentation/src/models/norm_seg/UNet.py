import torch
import torch.nn as nn
from torch.nn import functional as F

from tta_uia_segmentation.src.models.norm_seg.utils import get_conv, get_batch_norm, get_max_pool

from tta_uia_segmentation.src.utils.utils import assert_in

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, activation=nn.ReLU, n_dimensions=3):
        super().__init__()

        self.network = nn.Sequential(
            get_conv(in_channels, out_channels, 3, padding='same', bias=bias, n_dimensions=n_dimensions),
            get_batch_norm(out_channels, n_dimensions=n_dimensions),
            activation(inplace=True),
            get_conv(out_channels, out_channels, 3, padding='same', bias=bias, n_dimensions=n_dimensions),
            get_batch_norm(out_channels, n_dimensions=n_dimensions),
            activation(inplace=True),
        )

    def forward(self, x):
        return self.network(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_dimensions=3):
        super().__init__()

        self.double_conv = DoubleConv(in_channels, out_channels, n_dimensions=n_dimensions)
        self.downsample = get_max_pool(2, stride=2, n_dimensions=n_dimensions)

    def forward(self, x):
        x = self.double_conv(x)
        return x, self.downsample(x)


class Encoder(nn.Module):
    def __init__(self, channels=[1, 16, 32], n_dimensions=3):
        super().__init__()

        self.blocks = nn.ModuleList(
            [EncoderBlock(channels[i], channels[i+1], n_dimensions=n_dimensions)
             for i in range(len(channels)-1)]
        )

    def forward(self, x):
        skips = []

        for block in self.blocks:
            x_skip, x = block(x)
            skips.append(x_skip)

        return x, skips


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip, n_dimensions=3):
        super().__init__()
        
        assert_in(n_dimensions, 'n_dimensions', [1, 2, 3])

        if n_dimensions == 1:
            self.interpolation_mode = 'linear'
        elif n_dimensions == 2:
            self.interpolation_mode = 'bilinear'
        else:
            self.interpolation_mode = 'trilinear'

        self.skip = skip

        if skip:
            in_channels += out_channels

        self.double_conv = DoubleConv(
            in_channels, out_channels, n_dimensions=n_dimensions)

    def forward(self, x, skip):

        x = F.interpolate(
            x, skip.shape[2:], mode=self.interpolation_mode, align_corners=True)
        if self.skip:
            x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels=[128, 64, 32, 16], skips=[True, True, True], n_dimensions=3):
        super().__init__()

        assert len(channels) == len(skips) + 1

        self.blocks = nn.ModuleList(
            [DecoderBlock(channels[i], channels[i+1], skips[i], n_dimensions=n_dimensions)
             for i in range(len(channels)-1)]
        )

    def forward(self, x, skips):

        for block, x_skip in zip(self.blocks, skips):
            x = block(x, x_skip)

        return x


class UNet(nn.Module):
    """
    Convolutional UNet model in 1D, 2D or 3D
    
    Methods
    -------
    __init__(in_channels, n_classes, channels=[16, 32, 64], channels_bottleneck=128, skips=[True, True, True], n_dimensions=3)
        Initializes the UNet model
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        n_classes : int
            Number of output channels
        channels : list of int
            Number of channels in each block of the encoder and decoder
        channels_bottleneck : int
            Number of channels in the bottleneck block
        skips : list of bool
            Whether to use skip connections in each block of the decoder
        n_dimensions : int
            Type of layers to use, 1D, 2D or 3D
    """
    def __init__(self, in_channels, n_classes, channels=[16, 32, 64], channels_bottleneck=128, skips=[True, True, True], n_dimensions=3):
        super().__init__()

        assert len(channels) == len(skips), "channels and skips need to have same length"
        
        assert n_dimensions in [1, 2, 3], "n_dimensions must be 1, 2 or 3"

        self.encoder = Encoder([in_channels, *channels], n_dimensions=n_dimensions)
        self.conv_bottleneck = DoubleConv(channels[-1], channels_bottleneck, n_dimensions=n_dimensions)
        self.decoder = Decoder([channels_bottleneck, *channels[::-1]], skips, n_dimensions=n_dimensions)
        self.output_conv = get_conv(channels[0], n_classes, 1, n_dimensions=n_dimensions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x, skips = self.encoder(x)
        x = self.conv_bottleneck(x)
        x = self.decoder(x, skips[::-1])
        x = self.output_conv(x)

        return self.softmax(x), x


if __name__ == '__main__':
    net = UNet(1, 12, channels=[16, 32, 64], channels_bottleneck=128, skips=[True, True, True], two_dim=False)
    # print(net)
    x = torch.rand((8, 1, 40, 128, 128))
    y, logits = net(x)
    print(x.shape, y.shape)

    net = UNet(1, 12, channels=[16, 32, 64], channels_bottleneck=128, skips=[True, True, True], n_dimensions=3)
    # print(net)
    x = torch.rand((8, 1, 512, 512))
    y, logits = net(x)
    print(x.shape, y.shape)
