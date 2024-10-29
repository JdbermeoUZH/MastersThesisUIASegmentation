#!/bin/bash
#SBATCH --output=../../../../slurm_logs/%j_train_lcddpm.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g 
#SBATCH --gpus=1 --gres=gpumem:10g


if [ "$CLUSTER" = "bmic" ]; then
    source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
    conda activate $ENV_DIR
    echo "Python environment activated"
elif [ "$CLUSTER" = "euler" ]; then
    # Load necessary modules
    #source /cluster/project/cvl/admin/cvl_settings_ubuntu
    module load stack/2024-06 python_cuda/3.11.6 2>/dev/null

    # Activate python environment
    source $ENV_DIR/bin/activate

    # Copy data to compute node
    rsync -aqr $DATA_DIR ${TMPDIR}
    export DATA_DIR=${TMPDIR}/data
    echo "Data dir: $DATA_DIR"
    
    # Copy models necessary for training
    mkdir ${TMPDIR}/models
    rsync -aqr ${MODEL_DIR}/segmentation ${TMPDIR}/models/
    export MODEL_DIR=${TMPDIR}/models
    echo "Model dir: $MODEL_DIR"

    # Copy repo to compute node
    rsync -aqr $REPO_DIR ${TMPDIR}
    export REPO_DIR=${TMPDIR}/${REPO_DIR##*/}
    echo "Repo dir: $REPO_DIR"
    
    # Create new env variable in compute node for results
    export OLD_RESULTS_DIR=$RESULTS_DIR
    export RESULTS_DIR=${TMPDIR}/results
    echo "Results dir: $RESULTS_DIR"

    # Move to the dir where current script should be
    cd $REPO_DIR/scripts/training/ddpm
else
    echo "Python environment not activated. (env variable cluster: $CLUSTER)"
fi


python train_lcddpm.py \
    $REPO_DIR/config/datasets.yaml \
    $REPO_DIR/config/models.yaml \
    $REPO_DIR/config/training/training_hcp_t1.yaml \
    "$@" 


if [ "$CLUSTER" = "euler" ]; then
    rsync -auq ${TMPDIR}/ $LS_SUBCWD
    rsync -auq ${RESULTS_DIR}/ $OLD_RESULTS_DIR
fi

