#!/bin/bash
#SBATCH --output=../logs/preprocessing_bias_correction_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

cd ../tta_uia_segmentation/src/preprocessing
source /itet-stor/jbermeo/net_scratch/conda/etc/profile.d/conda.sh
conda activate nnUnet_dev

python 01_bias_correction.py "$@" #-d DATASET_ID
