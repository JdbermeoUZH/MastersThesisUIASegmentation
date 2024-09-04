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


def export_images(x_original, x_norm, y_original, y_pred, n_classes=-1, n_slices=8,
                   output_dir=None, image_name=None, **kwargs):
    if output_dir is None or image_name is None:
        return

    if n_classes == -1:
        n_classes = max(y_original.max(), y_pred.max()).item()

    x_original = x_original.squeeze()
    x_norm = x_norm.squeeze()
    y_original = onehot_to_class(y_original).squeeze()
    y_pred = onehot_to_class(y_pred).squeeze()

    vols_to_visualize = [
        ('Original Image', x_original, 'image'),
        ('Normalized Image', x_norm, 'image'),
        ('Ground Truth', y_original, 'segmentation'),
        ('Prediction', y_pred, 'segmentation'),
        ('Prediction Errors', y_pred == y_original, 'error_map')
    ]

    vols_to_visualize += [
        (vol_name, vol, 'image' if vol_name[0].lower() == 'x' else 'segmentation')
          for vol_name, vol in kwargs.items()
    ]
    
    D, H, W = x_original.shape

    margin = round(D / (2 * n_slices))
    slices = torch.linspace(margin, D - margin, n_slices, dtype=int)
    n_rows = len(vols_to_visualize)

    plt.figure(figsize=(10 * n_slices / n_rows, 10))
    for i, idx in enumerate(slices):
        for vol_j, (vol_name, vol, vol_type) in enumerate(vols_to_visualize):
            ax = plt.subplot(n_rows, n_slices, i + vol_j * n_slices + 1) 

            if vol_type == 'image':    
                plt.imshow(vol[idx, :, :], cmap='gray', interpolation='none', vmin=vol.min(), vmax=vol.max())
            
            elif vol_type == 'segmentation':
                plt.imshow(vol[idx, :, :], vmin=0, vmax=n_classes, cmap='tab20', interpolation='none')

            elif vol_type == 'error_map':
                plt.imshow(vol[idx, :, :], interpolation='none', cmap=ListedColormap(['black', 'white']), vmin=0, vmax=1)
                            
            else:
                raise ValueError(f"Unknown volume type: {vol_type}")
            
            if i == 0:
                plt.ylabel(vol_name)
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
