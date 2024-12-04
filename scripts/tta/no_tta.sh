#!/bin/bash
#SBATCH --output=../logs/%j_no_tta.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source $REPO_DIR/scripts/utils/start_env_and_data.sh

python no_tta.py \
 $REPO_DIR/config/datasets.yaml \
 $REPO_DIR/config/tta/tta.yaml \
 "$@"  
