#!/bin/bash
#SBATCH --output=../logs/%j_dae_tta.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1

source /itet-stor/jbermeo/net_scratch/conda/etc/profile.d/conda.sh
conda activate nnUnet_dev

python dae_tta.py \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/tta.yaml \
 "$@"  
