#!/bin/bash
#SBATCH --output=../logs/%j_precompute_dino_features.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1

source $REPO_DIR/scripts/utils/start_env_and_data.sh

python precompute_dino_features.py \
    $REPO_DIR/config/datasets.yaml \
    $REPO_DIR/config/preprocessing/preprocessing_hcp.yaml \
    "$@" 
