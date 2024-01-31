#!/bin/bash
#SBATCH --output=../logs/%j_normalize_with_nn.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python 06_normalize_images_with_nn.py \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
    "$@" 
