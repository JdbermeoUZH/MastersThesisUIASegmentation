"""
This script resamples the images to a desired resolution

It is used to resample the different TOF-MRA images to the same resolution
"""
import os
import datetime
import logging
import argparse

import numpy as np
import nibabel as nib

#---------- paths & hyperparameters
# hardcode them for now. Later maybe replace them.
multi_proc                = True
n_threads                 = 2
voxel_size                = np.array([0.3, 0.3, 0.6]) # hyper parameters to be set
save_logs                 = True

path_to_logs              = '/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/logs/preprocessing/resampling'
path_to_dataset           = '/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentationintermediate_results/images_intermediate_folder/global_thresholded_99.5'
path_to_save_process_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN'
#----------

date_now = datetime.now().strftime("%Y%m%d-%H%M%S")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    filename=os.path.join(path_to_logs, f'{date_now}_resampling.log'), filemode='w')
log = logging.Logger('Resampling')


def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Resampling of images')
    parser.add_argument('--voxel_size', type=float, nargs='+', default=voxel_size)  
    parser.add_argument('--multi_proc', action='store_true', default=multi_proc)
    parser.add_argument('--n_threads', type=int, default=n_threads)
    parser.add_argument('path_to_dataset', type=str, default=path_to_dataset)
    parser.add_argument('path_to_save_process_data', type=str, default=path_to_save_process_data)   
    parser.add_argument('path_to_logs', type=str, default=path_to_logs)    
    
    return parser.parse_args()


def resample(nii_img, new_voxel_size):
    old_dims       = nii_img.header.get_data_shape()
    old_voxel_size = nii_img.header.get_zooms()
    assert len(old_dims) == len(old_voxel_size) == len(new_voxel_size), \
        "New voxel size has to have the same dimension as the old voxel size"
    
    # new shape due to voxel changing
    new_dimensions = [int(old_dims[i] * old_voxel_size[i]/new_voxel_size[i]) for i in range(len(new_voxel_size))]
    new_nii_img = nib.processing.conform(
        nii_img,
        voxel_size = new_voxel_size,
        out_shape = new_dimensions,
        order = 3, cval=0, orientation='LPS'
        )
        
    return new_nii_img.get_fdata(), new_nii_img.affine, new_nii_img.header


if __name__ == '__main__':
    
    args = preprocess_cmd_args()
    print(args)
    
    