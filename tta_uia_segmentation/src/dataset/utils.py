import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_to_onehot(class_image, n_classes=-1, class_dim=1):
    """
    class_dim: dimension where classes are added (0 if class_image is an image and 1 if class_image is a batch)
    """
    class_image = class_image.long()
    one_hot = F.one_hot(class_image, n_classes).byte()
    one_hot = one_hot.squeeze(class_dim).movedim(-1, class_dim)
    
    return one_hot


def onehot_to_class(onehot, class_dim=1, keepdim=True):
    return onehot.argmax(dim=class_dim, keepdim=keepdim)


def normalize_quantile(data, min_p=0, max_p=1.0, clip: bool = True):
    min = torch.quantile(data, min_p)
    max = torch.quantile(data, max_p)
    return normalize_min_max(data, min, max, clip=clip)


def normalize_min_max(data, min=None, max=None, scale: float = 1, clip: bool = True):
    if min is None:
        min = torch.min(data)
    if max is None:
        max = torch.max(data)

    if max == min:
        data = torch.zeros_like(data)
    else:
        data = (data - min) / (max - min)
    
    if clip:
        data = torch.clip(data, 0, 1)
    data = scale * data
    
    if scale == 255:
        data = data.to(torch.uint8)
    
    return data

def normalize_standardize(data, mean=None, std=None):
    if mean is None:
        mean = torch.mean(data)
    if std is None:
        std = torch.std(data)

    data = (data - mean) / std
    
    return data


def normalize(type: str, **kwargs):
    if type == 'min_max':
        return normalize_min_max(**kwargs)
    elif type == 'quantile':
        return normalize_quantile(**kwargs)
    elif type == 'standardize':
        return normalize_standardize(**kwargs)
    else:
        raise ValueError(f'Unknown normalization type: {type}')

    
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
