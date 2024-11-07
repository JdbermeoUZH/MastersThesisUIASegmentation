from typing import Optional, Any

import torch
import numpy as np
from scipy import ndimage
from skimage import filters, morphology

from .BaseSegDecorator import BaseSegDecorator
from ..BaseSeg import BaseSeg
from tta_uia_segmentation.src.utils.utils import default, assert_in, torch_to_numpy
from tta_uia_segmentation.src.utils.io import deep_get


def get_background_mask(
    images: np.ndarray, bg_suppression_opts: dict, labels: Optional[np.ndarray] = None
):

    mask_source = deep_get(bg_suppression_opts, "mask_source", default="ground_truth")
    assert_in(mask_source, "mask_source", ["ground_truth", "thresholding"])

    if mask_source == "ground_truth":
        assert (
            labels is not None
        ), "labels must be provided if mask_source is 'ground_truth'"
        assert (
            images.shape == labels.shape
        ), f"Unequal shapes: images.shape: {images.shape}, labels.shape: {labels.shape}"

        return labels == 0

    shape = images.shape
    images = images.reshape(-1, *shape[-2:])

    thresholding = deep_get(bg_suppression_opts, "thresholding", default="otsu")
    assert_in(
        thresholding,
        "thresholding",
        ["isodata", "li", "mean", "minimum", "otsu", "triangle", "yen"],
    )

    if thresholding == "isodata":
        threshold = filters.threshold_isodata(images)
    elif thresholding == "li":
        threshold = filters.threshold_li(images)
    elif thresholding == "mean":
        threshold = filters.threshold_mean(images)
    elif thresholding == "minimum":
        threshold = filters.threshold_minimum(images)
    elif thresholding == "otsu":
        threshold = filters.threshold_otsu(images)
    elif thresholding == "triangle":
        threshold = filters.threshold_triangle(images)
    elif thresholding == "yen":
        threshold = filters.threshold_yen(images)

    fg_mask = images > threshold

    hole_filling = deep_get(bg_suppression_opts, "hole_filling", default=True)
    if hole_filling:
        for i in range(fg_mask.shape[0]):
            fg_mask[i] = morphology.binary_dilation(fg_mask[i])
            fg_mask[i] = ndimage.binary_fill_holes(fg_mask[i])

    fg_mask = fg_mask.reshape(shape)

    bg_mask = ~fg_mask

    return bg_mask


def background_suppression(x, bg_mask, opts=None, bg_class=0):
    """
    Suppresses the background of an image by setting the background pixels to a fixed value or a random value
    between a minimum and maximum value.
    """
    device = x.device

    suppression_type = deep_get(opts, "type", default="none", suppress_warning=True)
    assert_in(
        suppression_type, "suppression_type", ["none", "fixed_value", "random_value"]
    )

    if suppression_type == "fixed_value":
        bg_value = deep_get(opts, "bg_value", default=0)
        x = torch.where(bg_mask.bool(), bg_value, x)
        # x[bg_mask] = bg_value

    elif suppression_type == "random_value":
        bg_value_min = deep_get(opts, "bg_value_min", default=-1)
        bg_value_max = deep_get(opts, "bg_value_max", default=1)
        b, c, h, w = x.shape
        bg_value = torch.empty(b, c, 1, 1).uniform_(bg_value_min, bg_value_max)
        bg_value = bg_value.repeat(1, 1, h, w).to(device)
        x = torch.where(bg_mask.bool(), bg_value, x)

    return x


class BGSuppresionDecorator(BaseSegDecorator):
    def __init__(self, base_seg: BaseSeg, bg_suppression_opts: dict, bg_class: int = 0):
        super().__init__(base_seg)
        self._bg_suppression_opts = bg_suppression_opts
        self._bg_class = bg_class

    def _preprocess_x(
        self,
        x,
        bg_suppresion_opts: Optional[dict] = None,
        bg_class: Optional[int] = None,
        **preprocess_kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Get preprocessed x and intermediate outputs from wrapped class
        x_preproc, intermediate_outputs = self._base_seg._preprocess_x(
            x, **preprocess_kwargs
        )

        # Apply background suppression
        # ----------------------------
        bg_suppresion_opts_: dict = default(
            bg_suppresion_opts, self._bg_suppression_opts
        )
        bg_class_: int = default(bg_class, self._bg_class)

        # Get the background mask
        bg_mask = get_background_mask(torch_to_numpy(x_preproc), bg_suppresion_opts_)

        # Apply background suppression
        x_preproc = background_suppression(
            x_preproc, bg_mask, bg_suppresion_opts, bg_class=bg_class
        )

        intermediate_outputs["bg_mask"] = bg_mask
        intermediate_outputs["x_preproc_bg_supp"] = x_preproc

        return x_preproc, intermediate_outputs
