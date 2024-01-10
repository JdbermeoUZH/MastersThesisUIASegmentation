"""
This script resamples the images to a desired resolution

It is used to resample the different TOF-MRA images to the same resolution
"""
import os
import tqdm
from datetime import datetime
import logging
import argparse
from pprint import pprint

import numpy as np
import nibabel as nib
import nibabel.processing
import skimage.transform as ski_trf


from utils import (
    get_USZ_filepaths,
    get_ADAM_filepaths,
    get_Laussane_filepaths
)


#---------- paths & hyperparameters
# hardcode them for now. Later maybe replace them.
multi_proc                = True
n_threads                 = 2
voxel_size                = np.array([0.3, 0.3, 0.6]) # hyper parameters to be set
save_logs                 = True

path_to_logs              = '/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/logs/preprocessing/resampling'
path_to_USZ_dataset       = '/scratch_net/biwidl319/jbermeo/data/raw/USZ'
path_to_ADAM_dataset      = '/scratch_net/biwidl319/jbermeo/data/raw/ADAM'
path_to_Laussane_tof      = '/scratch_net/biwidl319/jbermeo/data/raw/Lausanne/original_images'
path_to_Laussane_seg      = '/scratch_net/biwidl319/jbermeo/data/raw/Lausanne/skull_stripped_and_aneurysm_mask'
path_to_save_processed_data = '/scratch_net/biwidl319/jbermeo/data/preprocessed/0_resampled'
#----------

date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(path_to_logs, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    filename=os.path.join(path_to_logs, f'{date_now}_resampling.log'), filemode='w')
log = logging.Logger('Resampling')


def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Resampling of images')
    parser.add_argument('--voxel_size', type=float, nargs='+', default=voxel_size)  
    parser.add_argument('--dataset', type=str, choices=['USZ', 'ADAM', 'Laussane'], default='USZ')
    parser.add_argument('--multi_proc', action='store_true', default=multi_proc)
    parser.add_argument('--n_threads', type=int, default=n_threads)
    parser.add_argument('--path_to_save_processed_data', type=str, default=path_to_save_processed_data)   
    parser.add_argument('--path_to_logs', type=str, default=path_to_logs)    
    
    return parser.parse_args()


def resize_segmentation(segmentation:np.ndarray, new_shape: tuple[int, ...], order: int = 3):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    
    taken from: https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py#L22
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return ski_trf.resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = ski_trf.resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        
        return np.round(reshaped).astype(np.int8)


def resample(
    nii_img: nib.nifti1.Nifti1Image, 
    new_voxel_size: tuple[float, float, float],
    order: int = 3,
    is_segmenation: bool = False,
    ) -> nib.nifti1.Nifti1Image:
    """
    Resample a nifti image to a new voxel size with scikit-image

    Parameters
    ----------
    nii_img : nib.nifti1.Nifti1Image
        Image to resample
    new_voxel_size : tuple[float, float, float]
        New voxel size
    order : int, optional
        Order of the spline interpolation, by default 3
    is_segmenation : bool, optional
        Whether the image is a segmentation mask, by default False. If True, the interpolation is done either with
        nearest neighbor or each class is interpolated independently and only voxels with a probability > 0.5 are kept
    """
    old_dims       = nii_img.header.get_data_shape()
    old_voxel_size = nii_img.header.get_zooms()
  
    assert len(old_dims) == len(old_voxel_size) == len(new_voxel_size), \
        "New voxel size has to have the same dimension as the old voxel size"
    
    # New shape due to change is voxel size
    new_shape = [int(old_dims[i] * old_voxel_size[i]/new_voxel_size[i]) for i in range(len(new_voxel_size))]
  
    resize_fn = lambda x: resize_segmentation(x, new_shape, order) if is_segmenation \
        else ski_trf.resize(x, new_shape, order, mode='edge', anti_aliasing=False)
    
    img_array = nii_img.get_fdata()
    resized_img_array = resize_fn(img_array)
    
    new_nii_img = nib.Nifti1Image(resized_img_array, affine=nii_img.affine, header=nii_img.header)
        
    return new_nii_img


def resample_image_and_segmentation_mask(output_dir, scans_dict, voxel_size):
    os.makedirs(output_dir, exist_ok=True)
    
    # For now let's do it sequentially. Later we can parallelize it
    for img_id, img_dict in tqdm.tqdm(scans_dict.items()):
        log.info(f"Resampling scan {img_id}")
        img_output_dir = os.path.join(output_dir, img_id)
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Load the TOF scan
        tof_scan = nib.load(img_dict['tof'])
        
        # Resample the TOF scan
        resampled_tof_scan = resample(tof_scan, voxel_size)
        
        # Save the resampled TOF scan
        nib.save(resampled_tof_scan, os.path.join(img_output_dir, f'{img_id}_tof.nii.gz'))
        
        # If the scan has a segmentation mask, resample it
        if 'seg' in img_dict.keys():
            seg_mask = nib.load(img_dict['seg'])
            resampled_seg_mask = resample(seg_mask, voxel_size, is_segmenation=True)
            
            # Save the resampled segmentation mask
            nib.save(resampled_seg_mask, os.path.join(img_output_dir, f'{img_id}_seg.nii.gz'))
            
        log.info(f"Scan {img_id} resampled")


if __name__ == '__main__':
    
    args = preprocess_cmd_args()
    pprint(args)
    
    # Load dictionary with filepaths of the scans and their respective segmentation masks
    if args.dataset == 'USZ':
        scans_dict = get_USZ_filepaths(path_to_USZ_dataset, include_segmentation_masks=True)
    elif args.dataset == 'ADAM':
        scans_dict = get_ADAM_filepaths(path_to_ADAM_dataset, include_segmentation_masks=True)
    elif args.dataset == 'Laussane':
        scans_dict = get_Laussane_filepaths(path_to_Laussane_tof, path_to_Laussane_seg)
    else:
        raise ValueError(f"No function to load filepaths of Dataset {args.dataset} "
                        "has been implemented")
    
    # Create folder to save the resampled scans
    dataset_output_dir = os.path.join(args.path_to_save_processed_data, args.dataset)

    resample_image_and_segmentation_mask(
        output_dir=dataset_output_dir,
        scans_dict=scans_dict,
        voxel_size=args.voxel_size
        )
    