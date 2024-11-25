import os

# account = 'staff'
# base_command = f'sbatch --account={account} no_tta_wmh_w_synthseg_labels.sh --wandb_log False --split test --bg_supp_x_norm_eval False --bg_supression_type none --classes_of_interest 16'

# Inference params
# :====================================:
batch_size = 2
save_predicted_vol_as_nifti = True

# Datasets
# :====================================:
dataset_type = "wmh"
split = "test"
target_datasets = ["umc", "nuhs", "vu"]
classes_of_interest = []
classes_of_interest = [str(c) for c in classes_of_interest]

# Trained Models
# :====================================:
seg_models_path = {
    # "umc": (
    #     "$RESULTS_DIR/wmh/segmentation/umc/dino/base/bs_32_lr_1em4_grad_clip_1.0_hier_1",
    #     "$RESULTS_DIR/wmh/segmentation/umc/dino/base/bs_32_lr_1em4_grad_clip_1.0_hier_0",
    #     "$RESULTS_DIR/wmh/segmentation/umc/dino/base/hierarchichal_decoder/bs_32_lr_1em4_grad_clip_1.0_hier_1",
    #     "$RESULTS_DIR/wmh/segmentation/umc/dino/base/hierarchichal_decoder/bs_32_lr_1em4_grad_clip_1.0_hier_2"
    # ),
    "nuhs": (
        "$RESULTS_DIR/wmh/segmentation/nuhs/dino/base/bs_32_lr_1em4_grad_clip_1.0_hier_1",
        "$RESULTS_DIR/wmh/segmentation/nuhs/dino/base/bs_32_lr_1em4_grad_clip_1.0_hier_0",
        "$RESULTS_DIR/wmh/segmentation/nuhs/dino/base/hierarchichal_decoder/bs_32_lr_1em4_grad_clip_1.0_hier_1",
        "$RESULTS_DIR/wmh/segmentation/nuhs/dino/base/hierarchichal_decoder/bs_32_lr_1em4_grad_clip_1.0_hier_2"
    ),
    # "vu": (
    #     "$RESULTS_DIR/wmh/segmentation/vu/dino/large/dice_loss_smoothing_den_1em10_opt_param_kerem_bs_16",
    #     "$RESULTS_DIR/wmh/segmentation/vu/dino/large/dice_loss_smoothing_den_1em10_opt_param_kerem_bs_16_grad_acc_2_lr_0.0001",
    # ),
}

# Command format
# :====================================:
base_command = (
    "python no_tta.py "
    + "$REPO_DIR/config/datasets.yaml "
    + "$REPO_DIR/config/tta/tta.yaml "
    + f" --wandb_log False" 
    + f" --split {split}"
    + f" --batch_size {batch_size}"
    + f" --save_predicted_vol_as_nifti {save_predicted_vol_as_nifti}"
    + " --viz_interm_outs "
)

base_command += (
    f" --classes_of_interest {' '.join(classes_of_interest)}"
    if len(classes_of_interest) > 0
    else ""
)

log_dir_base_path = "trained_on_{source_dataset}/tta_on_{target_dataset}"
log_dir_base_path = os.path.join(
    os.environ["RESULTS_DIR"],
    dataset_type,
    "tta",
    log_dir_base_path,
    split,
    "noTTA",
    "{seg_model_exp}",
)


# Run the commands
# :====================================:

for source_dataset, seg_model_paths in seg_models_path.items():

    for seg_model_path in seg_model_paths:
        # use the last to elements in the path as seg_model_exp
        tree_level = seg_model_path.split(os.path.sep).index(source_dataset) + 1
        seg_model_exp = os.path.join(*seg_model_path.split(os.path.sep)[tree_level:])
        
        for target_dataset in target_datasets:    
            log_dir = log_dir_base_path.format(
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                seg_model_exp=seg_model_exp,
            )

            print("#" * 70)
            print("Source Dataset:", source_dataset)    
            print("Target Dataset:", target_dataset)
            print("Experiment:", seg_model_exp) 
            print("#" * 70)
            os.system(
                f"{base_command} --seg_dir {seg_model_path} --dataset {target_dataset} --logdir {log_dir}"
            )
