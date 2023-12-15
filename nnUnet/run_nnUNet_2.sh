#!/bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --mem=40G
#SBATCH --gres=gpu:1  # titan_xp, geforce_gtx_titan_x, geforce_rtx_2080_ti
#SBATCH --cpus-per-task=4
#SBATCH --constraint='titan_xp'#|geforce_gtx_titan_x'

export nnUNet_raw="/scratch_net/biwidl319/jbermeo/data/nnUNet_raw"
export nnUNet_preprocessed="/scratch_net/biwidl319/jbermeo/data/nnUNet_preprocessed"
export nnUNet_results="/scratch_net/biwidl319/jbermeo/data/nnUnet_results"

source /itet-stor/jbermeo/net_scratch/conda/etc/profile.d/conda.sh
conda activate py_39
cd /scratch_net/biwidl319/jbermeo/UIASegmentation/

nvidia-smi

CUDA_LAUNCH_BLOCKING=1 nnUNetv2_train "$@"
