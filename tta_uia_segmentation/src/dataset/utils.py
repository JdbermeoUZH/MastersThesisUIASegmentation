from typing import Optional

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
    if type == "min_max":
        return normalize_min_max(**kwargs)
    elif type == "quantile":
        return normalize_quantile(**kwargs)
    elif type == "standardize":
        return normalize_standardize(**kwargs)
    else:
        raise ValueError(f"Unknown normalization type: {type}")


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def transform_orientation(x, y, z, orientation: str) -> tuple[float, float, float]:
    if orientation == "DHW":
        return z, x, y
    elif orientation == "HWD":
        return x, y, z
    elif orientation == "WHD":
        return y, x, z
    elif orientation == "DWH":
        return x, z, y
    elif orientation == "HDW":
        return x, z, y
    elif orientation == "WDH":
        return y, z, x
    else:
        raise ValueError(
            f"Invalid orientation: {orientation}. Must be one of 'DHW', 'HWD', 'WHD', 'DWH', 'HDW', 'WDH'."
        )


def calculate_class_weights(
    labels: np.ndarray,
    n_classes: int,
    normalize: bool = True,
    classes_of_interest: list = None,
    background: int = 0,
    clip_classes_of_interest_at_factor: Optional[float] = None,
):

    if classes_of_interest is not None and len(classes_of_interest) > 0:
        new_labels = np.zeros_like(labels)

        # Map all clasess of interest (usually one) to class 2
        new_labels = new_labels + np.where(np.isin(labels, classes_of_interest), 2, 0)

        # Map all other classes that are not background to class 1
        new_labels = new_labels + np.where(
            np.logical_not(np.isin(labels, [background] + classes_of_interest)), 1, 0
        )

        labels = new_labels
        n_classes_old = n_classes
        n_classes = np.unique(labels).size

    class_counts = np.zeros(n_classes)
    class_counts += np.bincount(labels.flatten(), minlength=n_classes)
    class_freq = class_counts / class_counts.sum()

    class_weights = 1 / class_freq
    class_weights = np.nan_to_num(class_weights, nan=0, posinf=0)

    if (
        clip_classes_of_interest_at_factor is not None
        and classes_of_interest is not None
        and class_weights[2] / class_weights[1] > clip_classes_of_interest_at_factor
    ):
        class_weights[2] = class_weights[1] * clip_classes_of_interest_at_factor

    if normalize:
        class_weights /= class_weights.sum()

    if classes_of_interest is not None and len(classes_of_interest) > 0:
        new_class_weights = np.zeros(n_classes_old)
        new_class_weights[background] = class_weights[0]
        new_class_weights[classes_of_interest] = class_weights[2]

        classes_of_no_interest = [
            c
            for c in range(n_classes_old)
            if c not in [background] + classes_of_interest
        ]
        new_class_weights[classes_of_no_interest] = class_weights[1]
        class_weights = new_class_weights

    return class_weights


def grayscale_to_rgb(img: torch.Tensor) -> torch.Tensor:
    repeat = [1] * len(img.shape)
    repeat[-3] = 3
    return img.repeat(*repeat)


def rgb_to_grayscale(img, is_originally_gray_scale: bool = False) -> torch.Tensor:
    # Extract the R, G, B channels
    r, g, b = img[:, 0, ...], img[:, 1, ...], img[:, 2, ...]

    # Apply the grayscale formula
    if is_originally_gray_scale:
        grayscale_img = img
    else:
        grayscale_img = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # Add a channel dimension for consistency (1, H, W)
    grayscale_img = grayscale_img.unsqueeze(0)

    return grayscale_img


def ensure_nd(n_dims: int, *arrays: np.ndarray | torch.Tensor) -> tuple[np.ndarray | torch.Tensor]:
    array_list_nd = []
    for array in arrays:
        current_dims = len(array.shape)
        if current_dims < n_dims:
            # Add singleton dimensions at the beginning until reaching the desired number of dimensions
            expanded_array = array[(None,) * (n_dims - current_dims)]
            array_list_nd.append(expanded_array)
        elif current_dims == n_dims:
            # If already the target number of dimensions, keep as is
            array_list_nd.append(array)
        else:
            # Optionally, raise an error or handle cases where dimensions exceed the target
            raise ValueError(f"Array has too many dimensions: {current_dims} > {n_dims}")

    # Return a single array if only one input, otherwise a tuple
    if len(array_list_nd) == 1:
        return array_list_nd[0]

    return tuple(array_list_nd)