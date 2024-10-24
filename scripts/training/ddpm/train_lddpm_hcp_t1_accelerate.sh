#!/bin/bash
#SBATCH --output=../../logs/%j_train_lcddpm_w_accelerate.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --constraint='geforce_gtx_titan_x'

if [ "$CLUSTER" = "bmic" ]; then
    source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
    conda activate $ENV_DIR

elif [ "$CLUSTER" = "euler" ]; then
    source $ENV_DIR/bin/activate

    # Copy data to compute node
    rsync -aq ./ ${TMPDIR}

    rsync -aqr /cluster/work/cvl/jbermeo/data/hcp/ ${TMPDIR}/

    cd $TMPDIR

else
    echo "Python environment not activated. (env variable cluster: $CLUSTER)"
fi

accelerate launch train_lcddpm.py \
    $REPO_DIR/config/datasets.yaml \
    $REPO_DIR/config/models.yaml \
    $REPO_DIR/config/training/training_hcp_t1.yaml \
    "$@" 

if [ "$CLUSTER" = "euler" ]; then
    rsync -auq ${TMPDIR}/ $LS_SUBCWD
fi

