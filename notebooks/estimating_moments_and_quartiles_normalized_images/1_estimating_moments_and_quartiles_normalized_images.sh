#!/bin/bash
#SBATCH --output=../../logs/%j_train_segmentation_2D.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'
#SBATCH --account='student'

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python 1_estimating_moments_and_quartiles_normalized_images.py "$@"  