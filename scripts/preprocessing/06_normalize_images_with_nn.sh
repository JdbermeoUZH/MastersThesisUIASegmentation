#!/bin/bash
#SBATCH --output=../logs/%j_normalize_with_nn.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate ddpm

python 06_normalize_images_with_nn.py \
    $REPO_DIR/config/datasets.yaml \
    $REPO_DIR/config/preprocessing/preprocessing_hcp.yaml \
    "$@" 
