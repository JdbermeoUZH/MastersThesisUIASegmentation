#!/bin/bash
#SBATCH --output=../../logs/%j_train_cddpm.out
#SBATCH --ntasks=3
#SBATCH --gres=gpu:2

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

mpiexec -n 2 python ./train_cddpm_oai.py \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/models.yaml \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/training/training_hcp_t1w.yaml 
    "$@" 
