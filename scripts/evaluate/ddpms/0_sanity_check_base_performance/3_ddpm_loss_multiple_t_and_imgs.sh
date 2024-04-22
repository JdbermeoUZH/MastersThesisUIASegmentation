#!/bin/bash
#SBATCH --output=../../../logs/%j_evaluate_ddpm.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --constraint='geforce_gtx_titan_x|titan_xp'
#SBATCH --account='student'

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python -u 3_ddpm_loss_multiple_t_and_imgs.py "$@" 