import os
import sys
import math

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.join('..', '..'))
from tta_uia_segmentation.src.dataset.dataset_precomputed_dino_features import get_datasets
from tta_uia_segmentation.src.utils.io import load_config
from tta_uia_segmentation.src.utils.utils import torch_to_numpy
from tta_uia_segmentation.src.dataset.utils import onehot_to_class

plot_torch_img = lambda img_: plt.imshow(torch_to_numpy(img_.squeeze()), cmap='gray')
plot_torch_seg = lambda seg_: plt.imshow(torch_to_numpy(onehot_to_class(seg_).squeeze()), cmap='tab20', interpolation='none')


# Load Dataset and get example image
#:-------------------------------
dataset_name = 'umc'
split = 'train'
dino_model = 'base'

dataset_cfg = load_config('../../config/datasets.yaml')
dataset_cfg = dataset_cfg[dataset_name]

#dataset_cfg['paths_preprocessed_dino'] = {'train': "/scratch/jbermeo/data/wmh/umc/data_umc_2d_size_256_256_depth_48_res_1_1_3_from_0_to_10_dino_base_hier_2.hdf5"}


(dataset,) = get_datasets(
    dataset_name    = dataset_name,
    splits          = [split],
    paths_preprocessed = dataset_cfg['paths_preprocessed'],
    paths_original  = dataset_cfg['paths_original'],
    paths_preprocessed_dino = dataset_cfg['paths_preprocessed_dino'][dino_model],
    hierarchy_level = 2,
    resolution_proc = dataset_cfg['resolution_proc'],
    dim_proc        = dataset_cfg['dim'],
    n_classes       = dataset_cfg['n_classes'],
    )

img, seg, _ = dataset[2]
print("TEST RETRIEVING SINGLE ITEM")

print(f"Image features: {len(img)}")
print(f"Image features shapes: {[f.shape for f in img]}")
print(f"Segmentation shape: {seg.shape}")


# print("TEST Using with a DataLoader")

# from torch.utils.data import DataLoader

# dl = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

# dataset_size = len(dataset)

# num_imgs = 0
# for img, seg, *_ in dl:
#     num_imgs += img[0].shape[0]

# print(f"Number of images in DataLoader: {num_imgs}")
# print(f"Dataset size: {dataset_size}")
# assert num_imgs == dataset_size, "Number of images in DataLoader is not the same as the dataset size"

# print('Test for different height and rescale factors')
