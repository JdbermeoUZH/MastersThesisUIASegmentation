#!/bin/bash
#SBATCH --output=../../logs/%j_train_lcddpm_w_accelerate.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --constraint='geforce_gtx_titan_x'


# ------------------  Python Environment And Data ------------------
if [ "$CLUSTER" = "bmic" ]; then
    source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
    conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg
    export CUDA_LAUNCH_BLOCKING=1

    #conda activate $ENV_DIR
    #echo "Python environment activated. (ENV_DIR: $ENV_DIR)"

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

    # Create a triton cache dir env variable
    export TRITON_CACHE_DIR=${TMPDIR}/TRITON_CACHE_DIR

else
    echo "Python environment not activated. (env variable cluster: $CLUSTER)"
fi


# ------------------  Separate Command Line Arguments --------------------

# Initialize empty arrays to hold arguments for each command
accel_args=()
train_args=()

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --accel_*) 
            # Strip the --accel_ prefix before adding to accel_args
            stripped_arg="${1/--accel_/--}"  # Remove the prefix
            accel_args+=("$stripped_arg" "$2") 
            shift ;;  # Move to the next argument
        *) 
            train_args+=("$1") ;;  # All other arguments go to train_lcddpm.py
    esac
    shift
done

# Debugging: print parsed arguments (optional)
echo "Accelerate args: ${accel_args[@]}"
echo "Training args: ${train_args[@]}"

# ------------------  Script ------------------------------------------------

echo "Job ID: $SLURM_JOB_ID"
#echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
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

