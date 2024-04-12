import os
import sys
import random
import argparse


import yaml
import numpy as np
import nibabel as nib

sys.path.append(os.path.join('..', '..', 'tta_uia_segmentation', 'src'))

from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets


#---------- Default args
seed_def                    = 123
dataset_params_fp_def       = '/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml'
output_dir_def              = '/scratch_net/biwidl319/jbermeo/data/wmh_miccai/original_vols_as_nifti_files'
#----------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts the original volumes from the dataset to nifti files')
    parser.add_argument('dataset', type=str help='Dataset to use')
    parser.add_argument('--output_dir', type=str, default=output_dir_def, help='Path to save the nifti files'   )
    parser.add_argument('--dataset_params_fp', type=str, default=dataset_params_fp_def, help='Path to the dataset parameters')
    parser.add_argument('--seed', type=int, default=seed_def, help='Seed for reproducibility')
    args = parser.parse_args()
    
    output_dir  = f'/scratch_net/biwidl319/jbermeo/data/wmh_miccai/{args.dataset}/original_vols_as_nifti_files'

    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_params = yaml.safe_load(open(args.dataset_params_fp, 'r'))
        
    (ds_train, ds_val, ds_test) = get_datasets(
        splits          = ['train', 'val', 'test'],
        paths           = dataset_params['paths_processed'],
        paths_original  = dataset_params['paths_original'], 
        image_size      = (48, 256, 256),
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
    
    print('Output directory:', output_dir)
    
    for ds_name, ds in ds_dict.items():
        print(f'Processing {ds_name} dataset')
        out_dir = os.path.join(output_dir, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        
        out_dir_imgs = os.path.join(out_dir, 'imgs')
        os.makedirs(out_dir_imgs, exist_ok=True)
        
        out_dir_labels = os.path.join(out_dir, 'labels')
        os.makedirs(out_dir_labels, exist_ok=True)
        
        for i in range(len(ds)):
            shape = ds.get_original_images(i)[0].shape
            if shape[-2] != 240 or shape[-1] != 240:
                print(i, shape)
            
            original_pix_size = ds.pix_size_original[:, i] # px, py, pz
            
            img, label, _ = ds.get_original_images(i, as_onehot=False)
            img = img.permute(0, 2, 3, 1).squeeze().numpy()
            label = label.permute(0, 2, 3, 1).squeeze().numpy()
            
            # save image as nifti fil
            affine = np.eye(4)
            affine[0,0] = original_pix_size[0]
            affine[1,1] = original_pix_size[1]
            affine[2,2] = original_pix_size[2]
            affine[1,:] = affine[1,:] * -1  # Flip y-axis
            img_nii = nib.Nifti1Image(img, affine)
            img_nii.header.set_zooms(original_pix_size)
            nib.save(img_nii, os.path.join(out_dir_imgs, f'vol_{i}.nii.gz'))
            
            # save label as nifti file
            label_nii = nib.Nifti1Image(label, np.eye(4))
            label_nii.header.set_zooms(original_pix_size)
            nib.save(label_nii, os.path.join(out_dir_labels, f'vol_{i}_label.nii.gz'))
    
    print('Done!')