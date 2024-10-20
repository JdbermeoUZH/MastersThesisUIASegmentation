from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tta_uia_segmentation.src.utils.loss import onehot_to_class


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


def plot_img_seg(img, seg, n_classes=14, img_title='Image', seg_title='Segmentation', figsize=(10, 5), intensity_range: tuple[float, float] = None):
    if seg.max() < 1.0:
        seg = seg * n_classes
    
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    v_min = img.min() if intensity_range is None else intensity_range[0]
    v_max = img.max() if intensity_range is None else intensity_range[1]
    plt.imshow(img, cmap='gray', interpolation='none', vmin=v_min, vmax=v_max)
    plt.xticks([])
    plt.yticks([])
    plt.title(img_title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0)
    plt.colorbar(cax=cax)
    
    ax = plt.subplot(1, 2, 2)
    plt.imshow(seg, vmin=0, vmax=n_classes, cmap='tab20', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.title(seg_title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0)
    plt.colorbar(cax=cax)
    
    plt.show()


def export_images(x_original, x_norm, y_original, y_pred, y_dae=None, x_guidance=None, 
                  n_classes=-1, n_slices=8, output_dir=None, image_name=None):
    if output_dir is None or image_name is None:
        return

    if n_classes == -1:
        n_classes = max(y_original.max(), y_pred.max()).item()

    x_original = x_original.squeeze()
    x_norm = x_norm.squeeze()
    x_guidance = x_guidance.squeeze() if x_guidance is not None else None
    y_original = onehot_to_class(y_original).squeeze()
    y_pred = onehot_to_class(y_pred).squeeze()
    y_dae = onehot_to_class(y_dae).squeeze() if y_dae is not None else None
    
    D, H, W = x_original.shape

    margin = round(D / (2 * n_slices))
    slices = torch.linspace(margin, D - margin, n_slices, dtype=int)
    n_rows = 5
    n_rows += 1 if y_dae is not None else 0
    n_rows += 1 if x_guidance is not None else 0 

    plt.figure(figsize=(10 * n_slices / n_rows, 10))
    for i, idx in enumerate(slices):
        ax=plt.subplot(n_rows, n_slices, i + 1)
        plt.imshow(x_original[idx, :, :], cmap='gray', interpolation='none', vmin=x_original.min(), vmax=x_original.max())
        if i == 0:
            plt.ylabel('Original Image')
        plt.xticks([])
        plt.yticks([])
        if i == n_slices - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            plt.colorbar(cax=cax)

        ax=plt.subplot(n_rows, n_slices, i + n_slices + 1)
        plt.imshow(x_norm[idx, :, :], cmap='gray', interpolation='none', vmin=x_norm.min(), vmax=x_norm.max())     
        if i == 0:
            plt.ylabel('Normalized Image')
        plt.xticks([])
        plt.yticks([])
        if i == n_slices - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            plt.colorbar(cax=cax)

        ax=plt.subplot(n_rows, n_slices, i + 2*n_slices + 1)
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

        ax=plt.subplot(n_rows, n_slices, i + 3*n_slices + 1)
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

        ax=plt.subplot(n_rows, n_slices, i + 4*n_slices + 1)
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
        
        if x_guidance is not None:
            ax=plt.subplot(n_rows, n_slices, i + 5 * n_slices + 1)
            plt.imshow(x_guidance[idx, :, :], cmap='gray', interpolation='none',
                       vmin=x_guidance.min(), vmax=x_guidance.max())
            if i == 0:
                plt.ylabel('Vol sampled from DDPM')
            plt.xticks([])
            plt.yticks([])
            if i == n_slices - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                plt.colorbar(cax=cax)
                
        if y_dae is not None:
            col_factor = 5 if x_guidance is None else 6

            ax=plt.subplot(n_rows, n_slices, i + col_factor * n_slices + 1)
            plt.imshow(y_dae[idx, :, :], vmin=0, vmax=14,
                        cmap='tab20', interpolation='none')
            if i == 0:
                plt.ylabel('DAE or Atlas')
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
