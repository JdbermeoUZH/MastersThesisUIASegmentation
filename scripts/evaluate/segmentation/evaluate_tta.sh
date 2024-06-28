#!/bin/bash
#SBATCH --output=../logs/%j_evaluate_seg.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg

python evaluate_tta.py \
 /scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/config/evaluate_tta.yaml \
 "$@"  

