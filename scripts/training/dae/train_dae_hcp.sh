#!/bin/bash
#SBATCH --output=../../logs/%j_train_dae_hcp.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1

cd ../tta_uia_segmentation/src/preprocessing
source /itet-stor/jbermeo/net_scratch/conda/etc/profile.d/conda.sh
conda activate nnUnet_dev

python train_dae_hcp.py \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/models.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/training_hcp_t1w.yaml\
 "$@"  
