#!/bin/bash
#SBATCH --output=../../../../slurm_logs/%j_train_lcddpm.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g 
#SBATCH --gpus=1 --gres=gpumem:10g


# ------------------  Python Environment And Data-- ---------------------

source $REPO_DIR/scripts/utils/start_env_and_data.sh


# ------------------  Script --------------------------------------------
if [ "$CLUSTER" = "euler" ]; then
    # Create a triton cache dir env variable
    export TRITON_CACHE_DIR=${TMPDIR}/TRITON_CACHE_DIR
    mkdir -p $TRITON_CACHE_DIR
fi

echo "Job ID: $SLURM_JOB_ID"

export OMP_NUM_THREADS=2

python train_cddpm.py \
    $REPO_DIR/config/datasets.yaml \
    $REPO_DIR/config/models.yaml \
    $REPO_DIR/config/training/training_hcp_t1.yaml \
    "$@" 


# ------------------  Copy Results From Compute Node --------------------

if [ "$CLUSTER" = "euler" ]; then
    rsync -auq ${TMPDIR}/ $LS_SUBCWD
    rsync -auq ${RESULTS_DIR}/ $OLD_RESULTS_DIR
fi

