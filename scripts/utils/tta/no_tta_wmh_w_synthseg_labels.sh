#!/bin/bash
#SBATCH --output=../logs/%j_no_tta.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='geforce_gtx_titan_x'

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python no_tta.py \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/tta/tta_wmh_umc_w_synthseg_labels.yaml \
 "$@"  
