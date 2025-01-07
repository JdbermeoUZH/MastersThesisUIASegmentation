#!/bin/bash
#SBATCH --output=../logs/%j_entropy_min.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='geforce_gtx_titan_x'

source $REPO_DIR/scripts/utils/start_env_and_data.sh

nvidia-smi

python tta_entropy_min.py \
 $REPO_DIR/config/datasets.yaml \
 $REPO_DIR/config/tta/tta.yaml \
 "$@"  

source $REPO_DIR/scripts/utils/clean_up.sh