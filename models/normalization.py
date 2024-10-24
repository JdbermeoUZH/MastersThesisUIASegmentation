import torch
import torch.nn as nn

from utils.io import deep_get
from utils.utils import assert_in


def background_suppression(x, bg_mask, opts=None, bg_class=0):
    """
    TODO: Most likely this function is not used anymore
    """
    device = x.device

    suppression_type = deep_get(opts, 'type', default='none', suppress_warning=True)
    assert_in(
        suppression_type, 'suppression_type',
        ['none', 'fixed_value', 'random_value']
    )

    if suppression_type == 'fixed_value':
        bg_value = deep_get(opts, 'bg_value', default=0)
        x[bg_mask] = bg_value

    elif suppression_type == 'random_value':
        bg_value_min = deep_get(opts, 'bg_value_min', default=-1)
        bg_value_max = deep_get(opts, 'bg_value_max', default=1)
        b,c,h,w = x.shape
        bg_value = torch.empty(b,c,1,1).uniform_(bg_value_min, bg_value_max)
        bg_value = bg_value.repeat(1,1,h,w).to(device)
        x[bg_mask] = bg_value[bg_mask]

    return x


class RBF(nn.Module):
    """
    Alternative to ReLU activation function
    """
    def __init__(self, n_channels, mean=0.2, stddev=0.05):
        super().__init__()

        self.mean = mean
        self.stddev = stddev

        self.scale = nn.Parameter(torch.empty((1, n_channels, 1, 1)), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.scale.normal_(self.mean, self.stddev)

    def forward(self, x):
        y = torch.exp(-(x**2) / (self.scale**2))
        return y


class Normalization(nn.Module):
    """
    Shallow network for normalization of images
    
    This network is the one that will be finetunned during Test-time adaptation instead of batch normalization layers.
    This will adapt/acount for the measurement shifts between the source and target domains.
    
    
    """
    def __init__(self, n_layers=3, image_channels=1, channel_size=16, kernel_size=3, activation=None, batch_norm=True, residual=True):
        super().__init__()

        assert n_layers > 0, "Normalization network has no layers"

        self.residual = residual

        channel_sizes = [image_channels] + (n_layers - 1) * [channel_size]

        layers = []
        for in_size, out_size in zip(channel_sizes, channel_sizes[1:]):
            layers += [nn.Conv3d(in_size, out_size, kernel_size,
                                 padding='same', padding_mode='reflect')]
            if batch_norm:
                layers += [nn.BatchNorm3d(out_size)]

            if activation == 'relu':
                layers += [nn.ReLU()]
            elif activation == 'elu':
                layers += [nn.ELU()]
            elif activation == 'rbf':
                layers += [RBF(out_size)]

        layers += [nn.Conv3d(channel_sizes[-1], image_channels,
                             kernel_size, padding='same', padding_mode='reflect')]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.layers(x)

        if self.residual:
            out += x
        
        return out


if __name__ == '__main__':
    model = Normalization(activation='rbf')
    print(model)
    x = torch.rand((8, 1, 128, 128))
    y = model(x)
    