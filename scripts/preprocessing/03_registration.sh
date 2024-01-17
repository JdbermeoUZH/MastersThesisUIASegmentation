#!/bin/bash
#SBATCH --output=../logs/%j_preprocessing_registration.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1

cd ../tta_uia_segmentation/src/preprocessing
source /itet-stor/jbermeo/net_scratch/conda/etc/profile.d/conda.sh
conda activate nnUnet_dev

python 03_registration.py \
    --multi_proc \
    --fixed_image_path /scratch_net/biwidl319/jbermeo/data/preprocessed/1_resampled/USZ/10015183-MCA-new/10015183-MCA-new_tof.nii.gz \
    --n_threads 4 \
    "$@" 
