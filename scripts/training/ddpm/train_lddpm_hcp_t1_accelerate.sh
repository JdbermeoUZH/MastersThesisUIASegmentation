#!/bin/bash
#SBATCH --output=../../logs/%j_train_lcddpm_w_accelerate.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --constraint='geforce_gtx_titan_x'


if [ "$CLUSTER" = "bmic" ]; then
    source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
    conda activate $ENV_DIR

elif [ "$CLUSTER" = "euler" ]; then
    source /cluster/project/cvl/admin/cvl_settings_ubuntu
    
    source $ENV_DIR/bin/activate

    # Copy data to compute node
    rsync -aqr $DATA_DIR/subcortical_structures/hcp/ ${TMPDIR}/data

    # Copy repo to compute node
    rsync -aqr $REPO_DIR ${TMPDIR}
    
    # Reassign the necessary env variables
    export DATA_DIR=${TMPDIR}/data/hcp
    echo "Data dir: $DATA_DIR"

    export REPO_DIR=${TMPDIR}/${REPO_DIR##*/}
    echo "Repo dir: $REPO_DIR"
    
    export OLD_RESULTS_DIR=$RESULTS_DIR
    export RESULTS_DIR=${TMPDIR}/results
    echo "Results dir: $RESULTS_DIR"

    # Move to the dir where current script should be
    cd $REPO_DIR/scripts/training/ddpm

else
    echo "Python environment not activated. (env variable cluster: $CLUSTER)"
fi

export OMP_NUM_THREADS=4

accelerate launch train_lcddpm.py \
    $REPO_DIR/config/datasets.yaml \
    $REPO_DIR/config/models.yaml \
    $REPO_DIR/config/training/training_hcp_t1.yaml \
    "$@" 

if [ "$CLUSTER" = "euler" ]; then
    rsync -auq ${TMPDIR}/ $LS_SUBCWD
    rsync -auq ${RESULTS_DIR}/ $OLD_RESULTS_DIR
fi

