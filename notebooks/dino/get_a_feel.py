import os
import sys
import math

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.join('..', '..'))
from tta_uia_segmentation.src.models.dino.DinoV2FeatureExtractor import DinoV2FeatureExtractor
from tta_uia_segmentation.src.models.dino.DinoSeg import DinoSeg, ResNetDecoder
from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets
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
    image_size      = [1, 256, 256],
    resolution_proc = dataset_cfg['resolution_proc'],
    dim_proc        = dataset_cfg['dim'],
    n_classes       = dataset_cfg['n_classes'],
    load_original   = False,
    )

img, seg, *_ = dataset[120]
img = img.unsqueeze(0)
seg = seg.unsqueeze(0).float()

print(f"Image shape: {img.shape}")
print(f"Segmentation shape: {seg.shape}")

# Check outputs of Dino
#:-------------------------------
img = img.repeat(1, 3, 1, 1)
dino_fe = DinoV2FeatureExtractor("large")

img = img.to('cuda')

dino_fe = dino_fe.to('cuda')

img_enc = dino_fe(img)['patch']
img_enc = img_enc.permute(0, 3, 1, 2)

# Define Resnet Decoder
#:-------------------------------
num_upsampling = math.ceil(math.log2(dino_fe.patch_size)) + 1
num_channels = [int(dino_fe.emb_dim / (2 ** i)) for i in range(num_upsampling)]

decoder = ResNetDecoder(output_size=[256, 256], n_classes=dataset_cfg['n_classes'], channels=num_channels, n_dimensions=2)
decoder = decoder.to('cuda') 

breakpoint()
decoder(img_enc)

dino_seg = DinoSeg(decoder, dino_fe, precalculated_fts=False)

breakpoint()
print("Done")
