#!/bin/bash
#SBATCH --output=../logs/%j_preprocessing_bias_correction_.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

cd ../tta_uia_segmentation/src/preprocessing
source /itet-stor/jbermeo/net_scratch/conda/etc/profile.d/conda.sh
conda activate nnUnet_dev

python 02_resampling.py "$@" 
