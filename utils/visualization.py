from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from utils.loss import onehot_to_class


def imshow(img, cmap='gray'):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show()


def multilabel_scatter(x, y, c=[], label=[], **kwargs):
    """
    Creates a scatter plot where each color is assigned its own label.
    """
    unique_colors = np.unique(c)
    assert len(label) == len(unique_colors)

    for lab, color in zip(label, unique_colors):
        color_mask = (c == color)
        plt.scatter(x[color_mask], y[color_mask], c=c[color_mask], label=lab, **kwargs)


def export_images(x_original, x_norm, y_original, y_pred, n_classes=-1, n_slices=8, output_dir=None, image_name=None):
    if output_dir is None or image_name is None:
        return

    if n_classes == -1:
        n_classes = max(y_original.max(), y_pred.max()).item()

    x_original = x_original.squeeze()
    x_norm = x_norm.squeeze()
    y_original = onehot_to_class(y_original).squeeze()
    y_pred = onehot_to_class(y_pred).squeeze()

    D, H, W = x_original.shape

    margin = round(D / (2 * n_slices))
    slices = torch.linspace(margin, D - margin, n_slices, dtype=int)

    plt.figure(figsize=(10*n_slices/5, 10))
    for i, idx in enumerate(slices):
        ax=plt.subplot(5, n_slices, i + 1)
        plt.imshow(x_original[idx, :, :], cmap='gray', interpolation='none', vmin=x_original.min(), vmax=x_original.max())
        if i == 0:
            plt.ylabel('Original Image')
        plt.xticks([])
        plt.yticks([])
        if i == n_slices - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            plt.colorbar(cax=cax)


        ax=plt.subplot(5, n_slices, i + n_slices + 1)
        plt.imshow(x_norm[idx, :, :], cmap='gray', interpolation='none', vmin=x_norm.min(), vmax=x_norm.max())
        if i == 0:
            plt.ylabel('Normalized Image')
        plt.xticks([])
        plt.yticks([])
        if i == n_slices - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            plt.colorbar(cax=cax)

        ax=plt.subplot(5, n_slices, i + 2*n_slices + 1)
        plt.imshow(y_original[idx, :, :], vmin=0, vmax=14,
                cmap='tab20', interpolation='none')
        if i == 0:
            plt.ylabel('Ground Truth')
        plt.xticks([])
        plt.yticks([])
        if i == n_slices - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            plt.colorbar(cax=cax)

        ax=plt.subplot(5, n_slices, i + 3*n_slices + 1)
        plt.imshow(y_pred[idx, :, :], vmin=0, vmax=14,
                cmap='tab20', interpolation='none')
        if i == 0:
            plt.ylabel('Prediction')
        plt.xticks([])
        plt.yticks([])
        if i == n_slices - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            plt.colorbar(cax=cax)

        ax=plt.subplot(5, n_slices, i + 4*n_slices + 1)
        plt.imshow(y_pred[idx, :, :] == y_original[idx, :, :],
                interpolation='none', cmap=ListedColormap(['black', 'white']), vmin=0, vmax=1)
        if i == 0:
            plt.ylabel('Prediction Errors')
        plt.xticks([])
        plt.yticks([])
        if i == n_slices - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            plt.colorbar(cax=cax)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close()
