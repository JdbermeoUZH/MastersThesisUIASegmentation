#!/bin/bash
#SBATCH --output=../logs/%j_dae_and_ddpm_tta.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'
#SBATCH --account='student'

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python dae_and_cddpm_tta.py \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/datasets.yaml \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/tta/tta_brain_hcp_t1w.yaml \
 "$@"  
#  --start 0 \
#  --stop 2 \

