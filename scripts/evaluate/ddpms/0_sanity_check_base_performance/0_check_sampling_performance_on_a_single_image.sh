#!/bin/bash
#SBATCH --output=../../../logs/%j_train_ddpm.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate ddpm

python 0_check_sampling_performance_on_a_single_image.py \
    --cpt_fp /scratch_net/biwidl319/jbermeo/logs/brain/ddpm/not_one_hot_128_base_filters/model-1-step_5000.pt \
    --params_fp /scratch_net/biwidl319/jbermeo/logs/brain/ddpm/not_one_hot_128_base_filters/params.yaml \
    --out_dir /scratch_net/biwidl319/jbermeo/results/ddpm/sanity_checks/num_filters_base_res_128/ \
    --exp_name cpt_1-step-5k/train \
    --split train \
    "$@" 
