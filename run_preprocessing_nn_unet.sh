#!/bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

export nnUNet_raw="/scratch_net/biwidl319/jbermeo/data/nnUNet_raw"
export nnUNet_preprocessed="/scratch_net/biwidl319/jbermeo/data/nnUNet_preprocessed"
export nnUNet_results="/scratch_net/biwidl319/jbermeo/data"

source /itet-stor/jbermeo/net_scratch/conda/etc/profile.d/conda.sh
conda activate UIASegmentation
nnUNetv2_plan_and_preprocess --verify_dataset_integrity "$@" #-d DATASET_ID
