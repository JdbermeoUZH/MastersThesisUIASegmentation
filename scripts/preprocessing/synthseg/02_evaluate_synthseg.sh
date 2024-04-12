#!/bin/bash
#SBATCH --output=../logs/%j_preprocessing_synthseg_labels.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'
#SBATCH --account='student'

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /itet-stor/jbermeo/net_scratch/conda_envs/synthseg_38_2

python 02_evaluate_synthseg.py "$@"  

