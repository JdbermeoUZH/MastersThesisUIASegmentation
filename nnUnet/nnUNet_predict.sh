#!/bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --mem=40G
#SBATCH --gres=gpu:1  # titan_xp, geforce_gtx_titan_x, geforce_rtx_2080_ti
#SBATCH --cpus-per-task=4

export nnUNet_raw="/scratch_net/biwidl319/jbermeo/data/nnUNet_raw"
export nnUNet_preprocessed="/scratch_net/biwidl319/jbermeo/data/nnUNet_preprocessed"
export nnUNet_results="/scratch_net/biwidl319/jbermeo/data/nnUnet_results"

source /itet-stor/jbermeo/net_scratch/conda/etc/profile.d/conda.sh
conda activate nnUnet_dev # pytcu11_8_py_3_9 - nn_unet_from_PyPi
cd /scratch_net/biwidl319/jbermeo/UIASegmentation/

nnUNet_n_proc_DA=0 nnUNetv2_predict --verbose -chk checkpoint_latest.pth -p 6GB_gpu -f all "$@"  #-i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities