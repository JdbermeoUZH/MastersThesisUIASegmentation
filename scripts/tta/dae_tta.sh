#!/bin/bash
#SBATCH --output=../logs/%j_dae_tta.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='geforce_gtx_titan_x'

source $REPO_DIR/scripts/utils/start_env_and_data.sh

nvidia-smi

# Build the command string with commands given in "$@"
command="python dae_tta.py \
  $REPO_DIR/config/datasets.yaml \
  $REPO_DIR/config/tta/tta.yaml \
  $@"

# Echo the command for logging
echo "Submitting job with command: $command"

# Run the command
$command

source $REPO_DIR/scripts/utils/clean_up.sh
