#!/bin/bash
#SBATCH --output=logs/%j_nnUnet_predict.out
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --constraint=GPUMEM32GB
#SBATCH --gres=gpu:V100:1

export nnUNet_raw="/home/jberme/scratch/nnUnet/data/raw"
export nnUNet_preprocessed="/home/jberme/scratch/nnUnet/data/nnUNet_preprocessed"
export nnUNet_results="/home/jberme/scratch/nnUnet/data/nnUNet_results"


module --ignore_cache load mamba
module --ignore_cache load gpu
module --ignore_cache load cuda
source activate nUnet

nvdia-smi
#nnUNetv2_predict --verbose -chk checkpoint_final.pth -f all "$@"  #-i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities

nnUNet_n_proc_DA=0 nnUNetv2_predict --verbose -chk checkpoint_latest.pth -f all "$@"  #-i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities