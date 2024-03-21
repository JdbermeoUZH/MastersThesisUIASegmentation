#!/bin/bash
#SBATCH --output=../../logs/%j_train_cddpm.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=3
#SBATCH --gres=gpu:1
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python train_cddpm.py \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/models.yaml \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/training/training_wmh_umc.yaml \
    "$@" 
