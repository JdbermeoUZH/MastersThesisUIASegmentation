import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

nn.Conv2d


def resize_to_multiple_of_patch(image, patch_size):
    """
    Resize the image to the ceiling multiple of the patch size by padding.

    Args:
        image (torch.Tensor): Image tensor of shape (N, C, H, W).
        patch_size (int): The patch size of the model.

    Returns:
        torch.Tensor: Resized image with height and width as multiples of patch_size.
    """
    h, w = image.shape[-2:]
    new_h = math.ceil(h / patch_size) * patch_size
    new_w = math.ceil(w / patch_size) * patch_size

    # Calculate the padding needed to make the dimensions multiples of patch_size
    pad_h = new_h - h
    pad_w = new_w - w

    pad_right = pad_w // 2
    pad_left = pad_w - pad_right

    pad_bottom = pad_h // 2
    pad_top = pad_h - pad_bottom

    # Apply padding to the bottom and right of the image
    padding = (pad_left, pad_right, pad_top, pad_bottom)  # (left, right, top, bottom)
    image_padded = F.pad(
        image, padding, mode="constant"
    )  # Reflect padding to avoid artifacts

    assert (
        image_padded.shape[-2] % patch_size == 0
        and image_padded.shape[-1] % patch_size == 0
    ), "The dimensions of the padded image are not multiples of the patch size."

    return image_padded


def resize_tensors_to_largest(tensors, mode="nearest"):
    """
    Resize a list of 3D tensors (H, W, C) to match the largest height and width among them.

    Args:
        tensors (list of torch.Tensor): A list of tensors to resize, where each tensor has shape (H, W, C).
        mode (str): Interpolation mode, 'bilinear' or 'nearest'.

    Returns:
        list of torch.Tensor: A list of resized tensors, all with the same shape.
    """
    # Extract shapes and find the largest height and width
    heights = [tensor.size(0) for tensor in tensors]
    widths = [tensor.size(1) for tensor in tensors]

    max_height = max(heights)
    max_width = max(widths)

    # Resize each tensor to (max_height, max_width)
    resized_tensors = []
    for tensor in tensors:
        tensor_resized = F.interpolate(
            tensor.permute(2, 0, 1).unsqueeze(0),
            size=(max_height, max_width),
            mode=mode,
        )
        resized_tensors.append(
            tensor_resized.squeeze(0).permute(1, 2, 0)
        )  # Convert back to original shape (H, W, C)

    return resized_tensors


normIN1K = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class DinoV2FeatureExtractor(nn.Module):

    def __init__(self, model="base_reg", inference_mode=True):
        self.model_name = model
        super(DinoV2FeatureExtractor, self).__init__()
        self.model = self._initialize_dino(model)
        self.model.eval()
        self._inference_mode = inference_mode

    def _forward(self, image, mask=None, pre=True, hierarchy: int = 0):

        if hierarchy > 0:
            image = F.interpolate(
                image, scale_factor=2**hierarchy, mode="bicubic", align_corners=False
            )

        patch_size = self.model.patch_size
        image = resize_to_multiple_of_patch(image, patch_size)

        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            mask = mask.flatten(1)  # N x hw

        output = {}
        nph, npw = image.size(2) // patch_size, image.size(3) // patch_size
        dino_out = self.model.forward_features(image, mask)

        output["cls"] = dino_out["x_norm_clstoken"]
        output["reg"] = dino_out["x_norm_regtokens"]
        patch = dino_out["x_norm_patchtokens"]
        # from N x hw x D -> N x h x w x D
        output["patch"] = patch.reshape(-1, nph, npw, patch.size(-1))
        output["mask"] = dino_out["masks"]

        if pre:
            pre_norm = dino_out["x_prenorm"]
            output["pre_cls"] = pre_norm[:, 0]
            output["pre_reg"] = pre_norm[:, 1 : self.model.num_register_tokens + 1]
            pre_patch = pre_norm[:, self.model.num_register_tokens + 1 :]
            output["pre_patch"] = pre_patch.reshape(-1, nph, npw, pre_patch.size(-1))

        return output

    def forward(self, image, mask=None, pre=True, hierarchy: int = 0):
        if self._inference_mode:
            with torch.inference_mode():
                breakpoint()
                return self._forward(image, mask, pre, hierarchy)
        else:
            return self._forward(image, mask, pre, hierarchy)

    def forward_hierarchically(self, image, mask=None, pre=True, hierarchy=[0, 1]):

        min_hier = np.min(hierarchy)

        patch_size = self.model.patch_size
        image = resize_to_multiple_of_patch(
            image, int(float(patch_size) * (2.0 ** (-float(min_hier))))
        )

        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            mask = mask.flatten(1)

        output = {}

        for hier in hierarchy:

            output_hier = {}

            resimg = image.clone()
            resimg = F.interpolate(
                resimg, scale_factor=2**hier, mode="bicubic", align_corners=False
            )

            h, w = resimg.size(2) // patch_size, resimg.size(3) // patch_size
            assert h * patch_size == resimg.size(2) and w * patch_size == resimg.size(3)

            dino_out = self.model.forward_features(resimg, mask)
            output_hier["cls"] = dino_out["x_norm_clstoken"]
            output_hier["reg"] = dino_out["x_norm_regtokens"]
            patch = dino_out["x_norm_patchtokens"]
            output_hier["patch"] = patch.reshape(-1, h, w, patch.size(-1))
            output_hier["mask"] = dino_out["masks"]

            if pre:
                pre_norm = dino_out["x_prenorm"]
                output_hier["pre_cls"] = pre_norm[:, 0]
                output_hier["pre_reg"] = pre_norm[
                    :, 1 : self.model.num_register_tokens + 1
                ]
                pre_patch = pre_norm[:, self.model.num_register_tokens + 1 :]
                output_hier["pre_patch"] = pre_patch.reshape(
                    -1, h, w, pre_patch.size(-1)
                )

            output[hier] = output_hier

        if len(hierarchy) == 1:
            return output[hierarchy[0]]

        return output

    def _initialize_dino(self, model) -> nn.Module:
        repo_or_dir = "facebookresearch/dinov2"
        if model == "small":
            model = torch.hub.load(repo_or_dir, "dinov2_vits14")
        elif model == "small_reg":
            model = torch.hub.load(repo_or_dir, "dinov2_vits14_reg")

        elif model == "base":
            model = torch.hub.load(repo_or_dir, "dinov2_vitb14")
        elif model == "base_reg":
            model = torch.hub.load(repo_or_dir, "dinov2_vitb14_reg")

        elif model == "large":
            model = torch.hub.load(repo_or_dir, "dinov2_vitl14")
        elif model == "large_reg":
            model = torch.hub.load(repo_or_dir, "dinov2_vitl14_reg")

        elif model == "giant":
            model = torch.hub.load(repo_or_dir, "dinov2_vitg14")
        elif model == "giant_reg":
            model = torch.hub.load(repo_or_dir, "dinov2_vitg14_reg")

        else:
            raise ValueError(f"Model {model} not supported.")

        return model

    @property
    def patch_size(self):
        return self.model.patch_size

    @property
    def emb_dim(self):
        return self.model.patch_embed.proj.out_channels

    @property
    def inference_mode(self):
        return self._inference_mode

    @inference_mode.setter
    def inference_mode(self, in_inference_mode):
        self._inference_mode = in_inference_mode

    def freeze(self):
        self.requires_grad_(False)


def patch2rgb(patch, maps=["umap", "tsne", "pca"]):

    h, w = patch.size(1), patch.size(2)

    patch = patch.reshape(-1, patch.shape[-1])
    patch = patch.detach().cpu().numpy()

    embRGBs = {}

    for map_i in maps:
        if map_i == "tsne":
            embedding = TSNE(n_components=3).fit_transform(patch)
            normalized_embedding = (embedding - embedding.min()) / (
                embedding.max() - embedding.min()
            )
            embRGB = normalized_embedding.reshape(h, w, 3)
            embRGBs["tsne"] = embRGB
        elif map_i == "pca":
            mapper = PCA(n_components=3).fit(patch)
            embedding = mapper.transform(patch)
            normalized_embedding = (embedding - embedding.min()) / (
                embedding.max() - embedding.min()
            )
            embRGB = normalized_embedding.reshape(h, w, 3)
            embRGBs["pca"] = embRGB

    return embRGBs


def visualize_rgbpatch(embRGBs, image=None, figsize=(15, 5)):

    add1 = 1 if image is not None else 0
    img = image.clone()

    plt.figure(figsize=figsize)

    if image is not None:
        plt.subplot(1, len(embRGBs) + 1, 1)
        plt.imshow(img.squeeze().permute(1, 2, 0))
        plt.axis("off")
        plt.title("Original Image")

    for i, key in enumerate(embRGBs):
        plt.subplot(1, len(embRGBs) + add1, i + add1 + 1)
        plt.imshow(embRGBs[key])
        plt.axis("off")
        plt.title(f"DINOv2 ({key})")

    plt.show()


def visualize_correlations(output, image, pre=False, cmap="jet"):

    img = image.clone()

    if not pre:
        npat = output["patch"]
        regs = output["reg"]
        regs = torch.concatenate([output["cls"].unsqueeze(1), regs], dim=1)
    else:
        npat = output["pre_patch"]
        regs = output["pre_reg"]
        regs = torch.concatenate([output["pre_cls"].unsqueeze(1), regs], dim=1)

    n_samples = npat.size(0)
    n_regs = regs.size(1)

    fig, axs = plt.subplots(n_samples, n_regs + 1, figsize=(15, 15))

    for i in range(n_samples):
        for jx in range(n_regs + 1):

            if jx == 0:
                if n_samples > 1:
                    axs[i, jx].imshow(img.squeeze().permute(1, 2, 0), cmap="gray")
                else:
                    axs[jx].imshow(img.squeeze().permute(1, 2, 0), cmap="gray")
                continue
            else:
                j = jx - 1

            p = npat[i].squeeze().detach().cpu().numpy()
            r = regs[i, j].squeeze().detach().cpu().numpy()

            p = p / np.linalg.norm(p)
            r = r / np.linalg.norm(r)

            m = np.sum(p * r, -1)

            if n_samples > 1:
                axs[i, jx].imshow(m, cmap=cmap)
            else:
                axs[jx].imshow(m, cmap=cmap)
