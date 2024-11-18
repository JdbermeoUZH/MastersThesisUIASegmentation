import os

# account = 'staff'
# base_command = f'sbatch --account={account} no_tta_wmh_w_synthseg_labels.sh --wandb_log False --split test --bg_supp_x_norm_eval False --bg_supression_type none --classes_of_interest 16'

# Inference params
# :====================================:
batch_size = 16

# Datasets
# :====================================:
dataset_type = "wmh"
split = "test"
target_datasets = ["umc_w_synthseg_labels", "nuhs_w_synthseg_labels", "vu_w_synthseg_labels"]
classes_of_interest = [16]
classes_of_interest = [str(c) for c in classes_of_interest]

# Trained Models
# :====================================:
model_type = "dino"  # 'dino' or 'norm_seg'

seg_models_path = {
    'umc_w_synthseg_labels': (
        "$RESULTS_DIR/wmh/segmentation/umc_w_synthseg_labels/dino/large/dice_loss_smoothing_den_1em10_opt_param_kerem_bs_16_grad_acc_2_lr_0.0001",
    ),
    'nuhs_w_synthseg_labels': (
        "$RESULTS_DIR/wmh/segmentation/nuhs_w_synthseg_labels/dino/large/dice_loss_smoothing_den_1em10_opt_param_kerem_bs_16_grad_acc_2_lr_0.0001",
    ),
    'vu_w_synthseg_labels': (
        "$RESULTS_DIR/wmh/segmentation/vu_w_synthseg_labels/dino/large/dice_loss_smoothing_den_1em10_opt_param_kerem_bs_16_grad_acc_2_lr_0.0001",
    )
}

# Command format
# :====================================:
base_command = (
    "python no_tta.py "
    + "$REPO_DIR/config/datasets.yaml "
    + "$REPO_DIR/config/tta/tta_brain_hcp_t1w.yaml "
    + f"--wandb_log False --split {split} --model_type {model_type}"
    + f" --batch_size {batch_size}"
    + " --viz_interm_outs "
    + f" --classes_of_interest {' '.join(classes_of_interest)}"
)

log_dir_base_path = "trained_on_{source_dataset}_TEST/tta_on_{target_dataset}"
log_dir_base_path = os.path.join(
    os.environ["RESULTS_DIR"],
    dataset_type,
    "tta",
    log_dir_base_path,
    split,
    'noTTA',
    model_type,
    "{seg_model_exp}",
)


# Run the commands
# :====================================:

for source_dataset, seg_model_paths in seg_models_path.items():

    for seg_model_path in seg_model_paths:
        # use the last to elements in the path as seg_model_exp
        seg_model_exp = os.path.join(
            *seg_model_path.split(os.path.sep)[-2:]
        )

        for target_dataset in target_datasets:
            log_dir = log_dir_base_path.format(
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                seg_model_exp=seg_model_exp,
            )
            
            os.system(
                f"{base_command} --seg_dir {seg_model_path} --dataset {target_dataset} --logdir {log_dir}"
            )
