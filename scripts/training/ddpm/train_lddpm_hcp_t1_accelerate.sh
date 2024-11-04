#!/bin/bash
#SBATCH --output=../../../../slurm_logs/%j_train_lcddpm.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g 
#SBATCH --gpus=4 --gres=gpumem:10g


# ------------------  Python Environment And Data ------------------

source $REPO_DIR/scripts/utils/start_env_and_data.sh


# ------------------  Separate Command Line Arguments --------------------

source $REPO_DIR/scripts/utils/separate_accelerate_cmd_line_args.sh

eval "$(parse_args "$@")"

echo "Accelerate args: ${accel_args[@]}"
echo "Training args: ${train_args[@]}"


# ------------------  Script ------------------------------------------------

# UNCOMMENT IF IT STARTS COMPLANING ABOUT THIS:
# Set Master address and port for parallel training
# if [ -z "${MASTER_ADDR}" ]; then
#     export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n1)
#     # basic form export MASTER_ADDR=$(hostname)
#     # Or for FQDN: export MASTER_ADDR=$(hostname -f)
#     # Or for SLURM: export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n1)
# fi

# # Similarly for MASTER_PORT
# if [ -z "${MASTER_PORT}" ]; then
#     export MASTER_PORT=29500
# fi
if [ "$CLUSTER" = "euler" ]; then
    # Create a triton cache dir env variable
    export TRITON_CACHE_DIR=${TMPDIR}/TRITON_CACHE_DIR
    mkdir -p $TRITON_CACHE_DIR
fi

echo "Job ID: $SLURM_JOB_ID"

export OMP_NUM_THREADS=2

accelerate launch "${accel_args[@]}" train_lcddpm.py \
    $REPO_DIR/config/datasets.yaml \
    $REPO_DIR/config/models.yaml \
    $REPO_DIR/config/training/training_hcp_t1.yaml \
    "${train_args[@]}" 

# ------------------  Copy Results From Compute Node --------------------
if [ "$CLUSTER" = "euler" ]; then
    rsync -auq ${TMPDIR}/ $LS_SUBCWD
    rsync -auq ${RESULTS_DIR}/ $OLD_RESULTS_DIR
fi

