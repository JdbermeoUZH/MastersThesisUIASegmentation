#!/bin/bash
#SBATCH --output=../../logs/%j_train_dae_umc_synthseg_labels.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python train_dae.py \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/models.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/training/training_wmh_umc_synthseg_labels.yaml \
 "$@"  
