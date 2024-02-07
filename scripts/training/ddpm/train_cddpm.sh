#!/bin/bash
#SBATCH --output=../../logs/%j_train_ddpm.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate ddpm

python train_cddpm.py \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/models.yaml \
    /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/training/training_hcp_t1w.yaml 
    "$@" 
