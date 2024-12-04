#!/bin/bash
#SBATCH --output=../../logs/%j_train_dino_seg.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='titan_xp|geforce_rtx_2080_ti|geforce_gtx_1080_ti|titan_x'

# Setup env and copy necessary files to compute node
source $REPO_DIR/scripts/utils/start_env_and_data.sh

python train_dino_seg.py \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/models.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/training/training.yaml \
 "$@"  

# Clean up (ie: remove wandb cache)
source $REPO_DIR/scripts/utils/clean_up.sh