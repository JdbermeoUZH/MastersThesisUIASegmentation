#!/bin/bash
#SBATCH --output=../../logs/%j_train_segmentation_2D.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1

source /itet-stor/jbermeo/net_scratch/conda/etc/profile.d/conda.sh
conda activate nnUnet_dev
cd /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/scripts/training/segmentation

python train_2D_segmentation_model.py \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/models.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/training_hcp_t1w.yaml\
 "$@"  
