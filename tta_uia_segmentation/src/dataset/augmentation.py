from typing import Optional

import numpy as np
import torch
from scipy.ndimage import map_coordinates, rotate, shift, gaussian_filter
from skimage import transform

from tta_uia_segmentation.src.utils.io import deep_get, number_or_list_to_array
from tta_uia_segmentation.src.utils.utils import (
    uniform_interval_sampling,
    torch_to_numpy,
    numpy_to_torch,
)


#
# adapted from https://github.com/neerakara/test-time-adaptable-neural-networks-for-domain-generalization/blob/master/utils.py#L238
# and https://github.com/neerakara/test-time-adaptable-neural-networks-for-domain-generalization/blob/master/utils.py#L296
#
def elastic_transform(image, label, sigma, alpha, rng):  # 3d

    D, H, W = image.shape

    # random_state.rand(*shape) generate an array of image size with random uniform noise between 0 and 1
    # random_state.rand(*shape)*2 - 1 becomes an array of image size with random uniform noise between -1 and 1
    # applying the gaussian filter with a relatively large std deviation (~20) makes this a relatively smooth deformation field, but with very small deformation values (~1e-3)
    # multiplying it with alpha (500) scales this up to a reasonable deformation (max-min:+-10 pixels)
    # multiplying it with alpha (1000) scales this up to a reasonable deformation (max-min:+-25 pixels)
    dx = (
        gaussian_filter((rng.random((H, W)) * 2 - 1), sigma, mode="constant", cval=0)
        * alpha
    )
    dy = (
        gaussian_filter((rng.random((H, W)) * 2 - 1), sigma, mode="constant", cval=0)
        * alpha
    )

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # save deformation field for all slices of the image
    for i in range(D):
        image[i, :, :] = map_coordinates(
            image[i, :, :],
            indices,
            order=1,
            mode="reflect",
        ).reshape(H, W)
    
        if label is not None:
            label[i, :, :] = map_coordinates(
                label[i, :, :],
                indices,
                order=0,
                mode="reflect",
            ).reshape(H, W)

    return image, label


#
# adapted from https://github.com/neerakara/test-time-adaptable-neural-networks-for-domain-generalization/blob/master/utils.py#L91
#
def crop_or_pad_slice_to_size(slice, nx, ny):
    z, x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        return slice[:, x_s : x_s + nx, y_s : y_s + ny]
    elif x <= nx and y > ny:
        return np.pad(
            slice[:, :, y_s : y_s + ny],
            [[0, 0], [x_c, nx - x_c - x], [0, 0]],
            mode="reflect",
        )
    elif x > nx and y <= ny:
        return np.pad(
            slice[:, x_s : x_s + nx, :],
            [[0, 0], [0, 0], [y_c, ny - y_c - y]],
            mode="reflect",
        )
    else:
        return np.pad(
            slice[:, :, :],
            [[0, 0], [x_c, nx - x_c - x], [y_c, ny - y_c - y]],
            mode="reflect",
        )


def get_augmentation_prob(tf_probs, augmentation_type: str, default_aug_prob) -> float:
    aug_prob = deep_get(
        tf_probs, augmentation_type, default=default_aug_prob, suppress_warning=True
    )

    assert (
        0 <= aug_prob <= 1
    ), f"Augmentation probability must be in [0, 1], got {aug_prob}"

    return aug_prob


#
# adapted from https://github.com/neerakara/test-time-adaptable-neural-networks-for-domain-generalization/blob/master/train_i2l_mapper.py#L588
#
def apply_data_augmentation(
    images: np.ndarray | torch.Tensor,
    labels: Optional[np.ndarray | torch.Tensor] = None,
    n_dimensions=2,
    da_ratio=0.25,
    p_inversion=0.0,
    inv_strength=0.8,
    tf_probs=None,
    sigma=20,
    alpha=1000,
    trans_min=-10,
    trans_max=10,
    rot_min=-10,
    rot_max=10,
    scale_min=0.9,
    scale_max=1.1,
    gamma_min=0.5,
    gamma_max=2.0,
    brightness_min=0.0,
    brightness_max=0.1,
    noise_mean=0.0,
    noise_std=0.1,
    rng=None,
    intensity_range=(0, 1),
    return_torch=False,
):

    images_ = (
        np.copy(images) if isinstance(images, np.ndarray) else torch_to_numpy(images)
    )
    assert isinstance(images_, np.ndarray) 

    if labels is not None:
        labels_ = (
            np.copy(labels) if isinstance(labels, np.ndarray) else torch_to_numpy(labels)
        )

        assert isinstance(labels_, np.ndarray)
    else:
        labels_ = None

    if rng is None:
        rng = np.random.default_rng()

    # ========
    # image inversion
    # ========
    if rng.random() < get_augmentation_prob(tf_probs, "image_inversion", p_inversion):
        images_ = 1 - inv_strength * images_

    # ========
    # elastic deformation
    # ========
    if rng.random() < get_augmentation_prob(tf_probs, "elastic_deformation", da_ratio):
        images_, labels_ = elastic_transform(
            images_, labels_, sigma=sigma, alpha=alpha, rng=rng
        )

    # ========
    # translation
    # ========
    if rng.random() < get_augmentation_prob(tf_probs, "translation", da_ratio):

        random_shift_x = rng.uniform(trans_min, trans_max)
        random_shift_y = rng.uniform(trans_min, trans_max)

        random_shifts = [random_shift_x, random_shift_y]
        if n_dimensions == 3:
            random_shift_z = rng.uniform(trans_min, trans_max)
            random_shifts = [random_shift_z] + random_shifts

        images_ = shift(images_, shift=(0, *random_shifts), order=1, mode="reflect")
        if labels_ is not None:
            labels_ = shift(labels_, shift=(0, *random_shifts), order=0, mode="reflect")

    # ========
    # rotation
    # ========
    if rng.random() < get_augmentation_prob(tf_probs, "rotation", da_ratio):
        random_angle = rng.uniform(rot_min, rot_max)

        images_ = rotate(
            images_,
            reshape=False,
            angle=random_angle,
            axes=(2, 1),
            order=1,
            mode="reflect",
        )
        if labels_ is not None:
            labels_ = rotate(
                labels_,
                reshape=False,
                angle=random_angle,
                axes=(2, 1),
                order=0,
                mode="reflect",
            )

    # ========
    # scaling
    # ========
    if rng.random() < get_augmentation_prob(tf_probs, "scaling", da_ratio):
        _, n_x, n_y = images_.shape
        scale_val = np.round(rng.uniform(scale_min, scale_max), 2)

        images_i_tmp = transform.rescale(
            images_,
            scale_val,
            order=1,
            preserve_range=True,
            mode="reflect",
            channel_axis=0,
        )
        
        images_ = crop_or_pad_slice_to_size(images_i_tmp, n_x, n_y)
        
        if labels_ is not None:        
            labels_i_tmp = transform.rescale(
                labels_,
                scale_val,
                order=0,
                preserve_range=True,
                mode="reflect",
                channel_axis=0,
            )

            labels_ = crop_or_pad_slice_to_size(labels_i_tmp, n_x, n_y)

    # ========
    # contrast
    # ========
    if rng.random() < get_augmentation_prob(tf_probs, "contrast", da_ratio):
        gamma_min = number_or_list_to_array(gamma_min)
        gamma_max = number_or_list_to_array(gamma_max)

        # gamma contrast augmentation
        c = np.round(uniform_interval_sampling(gamma_min, gamma_max, rng), 2)
        images_ = images_**c
        # not normalizing after the augmentation transformation,
        # as it leads to quite strong reduction of the intensity
        # range when done after high values of gamma augmentation

    # ========
    # brightness
    # ========
    if rng.random() < get_augmentation_prob(tf_probs, "brightness", da_ratio):
        brightness_min = number_or_list_to_array(brightness_min)
        brightness_max = number_or_list_to_array(brightness_max)

        # brightness augmentation
        c = np.round(uniform_interval_sampling(brightness_min, brightness_max, rng), 2)
        images_ = np.clip(
            images_ + c, a_min=intensity_range[0], a_max=intensity_range[1]
        )

    # ========
    # noise
    # ========
    if rng.random() < get_augmentation_prob(tf_probs, "noise", da_ratio):
        # noise augmentation
        n = np.random.normal(noise_mean, noise_std, size=images_.shape)
        images_ = images_ + n.astype(images_.dtype)

    if return_torch:
        images_ = numpy_to_torch(images_)
        labels_ = numpy_to_torch(labels_) if labels_ is not None else None

    return images_, labels_
