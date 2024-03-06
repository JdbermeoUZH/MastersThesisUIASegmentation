#!/bin/bash
#SBATCH --output=../../../logs/%j_train_ddpm.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python 2_sampling_performance_multiple_imgs_multiple_ts.py  \
    --num_workers 3 \
    "$@"
    #--exp_name cpt_3-step-15k/train \
    #--split train \
    #"$@"  

