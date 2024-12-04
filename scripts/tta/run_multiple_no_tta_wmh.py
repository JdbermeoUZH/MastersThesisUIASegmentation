
import os

# Logging params
# :====================================:
wandb_log = False

# Inference params
# :====================================:
slurm_jobs = True

batch_size = 16
num_workers = 3
save_predicted_vol_as_nifti = False
print_config = False

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
    #"umc": (
        # "$RESULTS_DIR/wmh/segmentation/umc/norm_seg/norm_k_3/bs_32_lr_1em4_grad_clip_1.0",
        # "$RESULTS_DIR/wmh/segmentation/umc/dino/large/hierarchichal_decoder/bs_32_lr_1em4_grad_clip_1.0_hier_2",
        # "$RESULTS_DIR/wmh/segmentation/umc/dino/large/hierarchichal_decoder/fg_only_loss_bs_32_lr_1em4_grad_clip_1.0_hier_2",
        # "$RESULTS_DIR/wmh/segmentation/umc/dino/large/hierarchichal_decoder/bigger_decoder_4x_bs_32_lr_1em4_grad_clip_1.0_hier_2",
        # "$RESULTS_DIR/wmh/segmentation/umc/dino/large/hierarchichal_decoder/bs_16_lr_1em3_NO_grad_clip_hier_2",
        # KEEP THIS ONE "$RESULTS_DIR/wmh/segmentation/umc/dino/large/hierarchichal_decoder/bs_16_lr_1em3_hier_2_128_64_32_16_adam",
        # KEEP THIS ONE "$RESULTS_DIR/wmh/segmentation/umc/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2",
        # KEEP THIS ONE "$RESULTS_DIR/wmh/segmentation/umc/dino/large/resnet_decoder/opt_params_kerem_bs_32_dice_loss_decay_hier_0",
        #"$RESULTS_DIR/wmh/segmentation/umc/dino/large/resnet_decoder/opt_params_kerem_bs_32_CE_loss_decay_hier_0",
        # KEEP THIS ONE "$RESULTS_DIR/wmh/segmentation/umc/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_0",
    #),
    # Last to be commented
    # "nuhs": (
    #      "$RESULTS_DIR/wmh/segmentation/nuhs/dino/large/hierarchichal_decoder/bs_16_lr_1em3_hier_2_128_64_32_16_adam",
    #      "$RESULTS_DIR/wmh/segmentation/nuhs/dino/large/hierarchichal_decoder/bs_32_lr_1em4_grad_clip_1.0_hier_2",
    #      "$RESULTS_DIR/wmh/segmentation/nuhs/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2",
    #      "$RESULTS_DIR/wmh/segmentation/nuhs/dino/norm_seg/norm_k_3/bs_16_lr_1em3_NO_grad_clip"
    # ),
    # "vu": (
    #      "$RESULTS_DIR/wmh/segmentation/vu/dino/large/hierarchichal_decoder/bs_16_lr_1em3_hier_2_128_64_32_16_adam",
    #      "$RESULTS_DIR/wmh/segmentation/vu/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2",
    #      "$RESULTS_DIR/wmh/segmentation/vu/dino/norm_seg/norm_k_3/bs_16_lr_1em3_NO_grad_clip",
    # ),
}

# Command format
# :====================================:
if slurm_jobs:
    account = "staff"
    gpu_type = None # "titan_xp"
    base_command = f"sbatch --account={account}"
    base_command += f" --constraint='{gpu_type}'" if gpu_type is not None else ""
    base_command += " no_tta.sh"
else:
    base_command = (
        "python no_tta.py "
        + "$REPO_DIR/config/datasets.yaml "
        + "$REPO_DIR/config/tta/tta.yaml "
    )

base_command += (
    f" --wandb_log {wandb_log}"
    + f" --split {split}"
    + f" --batch_size {batch_size}"
    + f" --num_workers {num_workers}"
    + f" --save_predicted_vol_as_nifti {save_predicted_vol_as_nifti}"
    + f" --print_config {print_config}"
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

