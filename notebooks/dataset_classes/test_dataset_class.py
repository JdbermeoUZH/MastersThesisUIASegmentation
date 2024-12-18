import os
import sys
import math

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.join('..', '..'))
from tta_uia_segmentation.src.dataset.dataset import get_datasets
from tta_uia_segmentation.src.utils.io import load_config
from tta_uia_segmentation.src.utils.utils import torch_to_numpy
from tta_uia_segmentation.src.dataset.utils import onehot_to_class

plot_torch_img = lambda img_: plt.imshow(torch_to_numpy(img_.squeeze()), cmap='gray')
plot_torch_seg = lambda seg_: plt.imshow(torch_to_numpy(onehot_to_class(seg_).squeeze()), cmap='tab20', interpolation='none')


# Load Dataset and get example image
#:-------------------------------
dataset_name = 'hcp_t1'
split = 'train'

dataset_cfg = load_config('../../config/datasets.yaml')
dataset_cfg = dataset_cfg[dataset_name]


(dataset,) = get_datasets(
    dataset_name    = dataset_name,
    splits          = [split],
    paths           = dataset_cfg['paths_processed'],
    paths_original  = dataset_cfg['paths_original'],
    resolution_proc = dataset_cfg['resolution_proc'],
    dim_proc        = dataset_cfg['dim'],
    n_classes       = dataset_cfg['n_classes'],
    load_original   = False,
    load_in_memory  = False,
    rescale_factor  = [1, 1, 0.25],
    rescale_mode    = "trilinear",
    rescale_only_inplane = False,
    mode            = '3D',
    orientation     = 'depth',   
    )

img, seg, *_ = dataset[2]

print(f"Image shape: {img.shape}")
print(f"Segmentation shape: {seg.shape}")


print('Test for different height and rescale factors')


# dataset.orientation = 'width' #WDH
# #dataset.rescale_factor = [0.5, 0.5, 1.0]
# img, seg, *_ = dataset[64]

# print(f"Image shape: {img.shape}")
# print(f"Segmentation shape: {seg.shape}")


# dataset.orientation = 'depth' # HWD
# #dataset.rescale_factor = [0.5, 0.5, 1.0]
# img, seg, *_ = dataset[1]

# print(f"Image shape: {img.shape}")
# print(f"Segmentation shape: {seg.shape}")

# # plt.imshow(img.squeeze(), cmap='gray')
# # plt.show()

# plt.imshow(onehot_to_class(seg).squeeze(), vmin=0, vmax=dataset_cfg['n_classes'] -1 , cmap="tab20", interpolation="none")
# plt.show()

dataset.orientation = 'depth' 
img, seg, *_ = dataset.get_original_image(
    index=128,
    output_format='CHW'
)
print(f"Image shape: {img.shape}")
print(f"Segmentation shape: {seg.shape}")

dataset.orientation = 'depth' 
img, seg, *_ = dataset.get_original_image(
    index=128,
    output_format='NCHW'
)
print(f"Image shape: {img.shape}")
print(f"Segmentation shape: {seg.shape}")

