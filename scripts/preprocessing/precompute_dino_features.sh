#!/bin/bash
#SBATCH --output=../logs/%j_precompute_dino_features.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python precompute_dino_features.py \
    $REPO_DIR/config/datasets.yaml \
    $REPO_DIR/config/preprocessing/preprocessing_hcp.yaml \
    "$@" 
