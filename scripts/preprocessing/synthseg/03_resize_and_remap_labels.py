import os
import sys
import glob
import json
import shutil
import random
import argparse
from collections import defaultdict

import yaml
import h5py
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc

sys.path.append(os.path.join('..', '..', 'tta_uia_segmentation', 'src'))

from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets


#---------- Default args
seed_def                    = 123
dataset_params_fp_def       = '/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml'
input_dir_def               = '/scratch_net/biwidl319/jbermeo/data/preprocessed/synthseg_predictions/on_peprocessed_vols/wmh'      
output_dir_def              = '/scratch_net/biwidl319/jbermeo/data/wmh_miccai'
image_size_def              = (1, 256, 256)
#----------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts the original volumes from the dataset to nifti files')
    parser.add_argument('dataset', type=str, help='Dataset to use')
    parser.add_argument('--input_dir', type=str, default=input_dir_def, help='Path to the synthseg predictions'   )
    parser.add_argument('--output_dir', type=str, default=output_dir_def, help='Path to save the nifti files'   )
    parser.add_argument('--dataset_params_fp', type=str, default=dataset_params_fp_def, help='Path to the dataset parameters')
    parser.add_argument('--image_size', type=int, nargs=3, default=image_size_def, help='Image size')
    parser.add_argument('--seed', type=int, default=seed_def, help='Seed for reproducibility')
    parser.add_argument('--use_original_imgs', action='store_true', help='Use preprocessed images')
    args = parser.parse_args()
    
    output_dir  = os.path.join(args.output_dir, args.dataset)

    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_params = yaml.safe_load(open(args.dataset_params_fp, 'r'))[args.dataset]
        
    (ds_train, ds_val, ds_test) = get_datasets(
        splits          = ['train', 'val', 'test'],
        paths           = dataset_params['paths_processed'],
        paths_original  = dataset_params['paths_original'], 
        image_size      = args.image_size,
        resolution_proc = dataset_params['resolution_proc'],
        dim_proc        = dataset_params['dim'],
        n_classes       = dataset_params['n_classes'],
        aug_params      = None,
        deformation     = None,
        load_original   = True,
        bg_suppression_opts = None
    )
    
    ds_dict = {
        'train': ds_train,
        'val': ds_val,
        'test': ds_test
    }
    
    label_remapping_dict = json.load(open(
        os.path.join(os.path.dirname(__file__), 'remap_classes.json'), 'r'))
    
    new_labels_to_old_labels = defaultdict(list)
    for old_label, new_label_info in label_remapping_dict.items():
        new_labels_to_old_labels[int(new_label_info['new_label'])].append(int(old_label))
    
    old_labels_set = set([float(old_label) for old_label in label_remapping_dict.keys()])
    num_new_fg_labels = len(set([new_label_info['new_label'] for new_label_info in label_remapping_dict.values()])) - 1
    
    print('Output directory:', output_dir)
    
    for ds_name, ds in ds_dict.items():
        # Load each predicted volume 
        vol_fps = sorted(glob.glob(
            os.path.join(args.input_dir, args.dataset, ds_name, '*_synthseg.nii.gz'))
        )
        
        voxel_size = ds.resolution_proc #np.array(ds.resolution_proc)[[2, 1, 0]]
        out_shape = np.array(ds.dim_proc)[[2, 1, 0]]

        # Resize and remap the labels
        resampled_and_remapped_vols = []
        for vol_fp in vol_fps:
            # Load the volume
            vol = nib.load(vol_fp)
            
            # Resize the volume to the original pixel dimension and size
            vol = nibproc.conform(
                vol, 
                voxel_size=voxel_size, 
                out_shape=out_shape, 
                order = 0, cval=0, orientation='RPS'
            ).get_fdata()
            
            # Remap the labels
            assert old_labels_set.issuperset(set(np.unique(vol))), \
                "The volume has labels that are not in the remapping dictionary"
            
            one_hot_vol = np.zeros((len(new_labels_to_old_labels), *vol.shape), dtype=np.uint8)
            for new_label, old_labels in new_labels_to_old_labels.items():
                one_hot_vol[new_label] = np.isin(vol, old_labels).astype(np.uint8)
            vol = one_hot_vol.argmax(axis=0)

            resampled_and_remapped_vols.append(vol)
    
        # Replace the labels in the dataset
        new_labels = np.array(resampled_and_remapped_vols)
        new_labels = new_labels.transpose((0, 3, 1, 2))
        new_labels = new_labels.reshape(-1, *new_labels.shape[2:])
        
        # Assert they have the same shape
        assert ds.labels.squeeze().shape == new_labels.shape, "The resampled and remapped labels have different shapes"    
        
        # Save the h5 file with the new labels
        orig_h5_path = ds.path if not args.use_original_imgs else ds.path_original
        filename = os.path.basename(orig_h5_path).removesuffix('.hdf5')
        new_h5_path = os.path.join(args.output_dir, args.dataset, f'{filename}_w_synthseg_labels.hdf5')
        shutil.copy(orig_h5_path, new_h5_path)

        with h5py.File(new_h5_path, 'r+') as data:
            # Overwrite synthseg labels with original non-zero labels
            original_labels = data['labels'][:]
            
            # Shift foreground labels by the number of new_labels
            num_original_fg_labels = len(set(np.unique(original_labels))) - 1
            original_labels = np.where(original_labels > 0, original_labels + num_new_fg_labels, original_labels)
            
            new_labels = np.where(original_labels == 0, new_labels, original_labels)
            
            # Replace the labels in the hdf5 file
            del data['labels']
            data.create_dataset('labels', data=new_labels)
            
        # Check the new labels were saved correctly
        with h5py.File(new_h5_path, 'r') as data:
            assert np.allclose(data['labels'][:], new_labels), "The new labels were not saved correctly"
            
            # Check the images are the same
            with h5py.File(orig_h5_path, 'r') as orig_data:
                assert np.allclose(data['images'][:], orig_data['images'][:]), "The images are not the same"
        
        
        
        