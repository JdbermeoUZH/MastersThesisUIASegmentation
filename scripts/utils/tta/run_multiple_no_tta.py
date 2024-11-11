import os

# account = 'staff'
# base_command = f'sbatch --account={account} no_tta_wmh_w_synthseg_labels.sh --wandb_log False --split test --bg_supp_x_norm_eval False --bg_supression_type none --classes_of_interest 16'

# Datasets
# :====================================:
dataset_type = "subcortical_structures"
source_dataset = "abide_stanford"
exclude_sd = False
split = "test"

datasets = ["hcp_t1", "hcp_2", "abide_caltech", "abide_stanford"]
target_datasets = (
    datasets if not exclude_sd else [ds for ds in datasets if ds != source_dataset]
)

# Model type
# :====================================:
model_type = "dino"  # 'dino' or 'norm_seg'

# Command format
# :====================================:
base_command = (
    "python no_tta.py "
    + "$REPO_DIR/config/datasets.yaml "
    + "$REPO_DIR/config/tta/tta_brain_hcp_t1w.yaml "
    + f"--wandb_log False --split {split} --model_type {model_type}"
    + " --viz_interm_outs "
)

log_dir_base_path = "trained_on_{source_dataset}/tta_on_{target_dataset}"
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


seg_models_path = {
    "large/dice_loss_smoothing_den_1em10": "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/dice_loss_smoothing_den_1em10",
    "large/dice_loss_smoothing_den_1em10_opt_param_kerem": "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/dice_loss_smoothing_den_1em10_opt_param_kerem",
    "base/dice_loss_no_smoothing_den_1em10_bs_16": "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/base/dice_loss_no_smoothing_den_1em10_bs_16",
    "base/dice_loss_no_smoothing_den_1em10_bs_32": "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/base/dice_loss_no_smoothing_den_1em10_bs_32",
}

# Run the commands
# :====================================:

for seg_model_exp, seg_model_dir in seg_models_path.items():
    for target_dataset in target_datasets:
        log_dir = log_dir_base_path.format(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            seg_model_exp=seg_model_exp,
        )

        os.system(
            f"{base_command} --seg_dir {seg_model_dir} --dataset {source_dataset} --logdir {log_dir}"
        )
