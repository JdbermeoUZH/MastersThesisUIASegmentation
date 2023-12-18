#!/bin/bash
#SBATCH --output=logs/%j_nnUnet_train_V100.out
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --constraint=GPUMEM16GB
#SBATCH --gres=gpu:V100:1

export nnUNet_raw="/home/jberme/scratch/nnUnet/data/raw"
export nnUNet_preprocessed="/home/jberme/scratch/nnUnet/data/nnUNet_preprocessed"
export nnUNet_results="/home/jberme/scratch/nnUnet/data/nnUNet_results"


module --ignore_cache load mamba
module --ignore_cache load gpu
module --ignore_cache load cuda
source activate pytcu11_8_py_3_9

nvidia-smi

#nnUNet_n_proc_DA=0 use this prefix to not do multithreading
nnUNetv2_train "$@" #DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz  # 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres 