#!/bin/bash
#SBATCH --output=../../logs/%j_train_lcddpm.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='geforce_gtx_titan_x'

if [ "$CLUSTER" = "bmic" ]; then
    source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
    conda activate $ENV_DIR

elif [ "$CLUSTER" = "euler" ]; then
    source $ENV_DIR

else
    echo "Python environment not activated. (env variable cluster: $CLUSTER)"
fi

python train_lcddpm.py \
    $REPO_DIR/config/datasets.yaml \
    $REPO_DIR/config/models.yaml \
    $REPO_DIR/config/training/training_hcp_t1.yaml \
    "$@" 
