import torch.nn as nn

from tta_uia_segmentation.src.utils.utils import assert_in


def get_conv(in_channels, out_channels, kernel_size, n_dimensions=3, *args, **kwargs):
    assert_in(n_dimensions, 'n_dimensions', [1, 2, 3])

    if n_dimensions == 1:
        conv = nn.Conv1d
    elif n_dimensions == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    return conv(in_channels, out_channels, kernel_size, *args, **kwargs)


def get_batch_norm(num_features, n_dimensions=3, *args, **kwargs):
    assert_in(n_dimensions, 'n_dimensions', [1, 2, 3])

    if n_dimensions == 1:
        batch_norm = nn.BatchNorm1d
    elif n_dimensions == 2:
        batch_norm = nn.BatchNorm2d
    else:
        batch_norm = nn.BatchNorm3d

    return batch_norm(num_features, *args, **kwargs)


def get_max_pool(kernel_size, n_dimensions=3, *args, **kwargs):
    assert_in(n_dimensions, 'n_dimensions', [1, 2, 3])

    if n_dimensions == 1:
        max_pool = nn.MaxPool1d
    elif n_dimensions == 2:
        max_pool = nn.MaxPool2d
    else:
        max_pool = nn.MaxPool3d

    return max_pool(kernel_size, *args, **kwargs)

