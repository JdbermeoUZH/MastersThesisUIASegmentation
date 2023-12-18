#!/bin/bash
#SBATCH --output=logs/%j_nnUnet_predict_cpu.out
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4


export nnUNet_raw="/home/jberme/scratch/nnUnet/data/raw"
export nnUNet_preprocessed="/home/jberme/scratch/nnUnet/data/nnUNet_preprocessed"
export nnUNet_results="/home/jberme/scratch/nnUnet/data/nnUNet_results"


module --ignore_cache load mamba
source activate nUnet

nvdia-smi
nnUNetv2_predict --verbose -chk checkpoint_latest.pth -f all -device cpu "$@"  #-i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities