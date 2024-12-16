import os

# Logging params
# :====================================:
wandb_log = False

# Inference params
# :====================================:
slurm_jobs = True

batch_size = 4
num_workers = 3
save_predicted_vol_as_nifti = False
print_config = False

# Datasets
# :====================================:
dataset_type = "subcortical_structures"
split = "test"
target_datasets = ["abide_stanford", "abide_caltech", "hcp_t1", "hcp_t2"]
# target_datasets = ["abide_stanford", "hcp_t2"] 
#target_datasets = ["abide_caltech", "hcp_t1"] 
classes_of_interest = []
classes_of_interest = [str(c) for c in classes_of_interest]

# Trained Models
# :====================================:
seg_models_path = {
    "abide_stanford": (
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_9",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_30",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_42",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_109",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_232",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_620",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_937",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/norm_seg/norm_k_3/bs_16_lr_1em3_NO_grad_clip",
        #"$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/opt_params_kerem_bs_32_dice_loss_decay_hier_0",
        #"$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/resnet_decoder/opt_params_kerem_bs_32_dice_loss_decay_hier_2_CHECK",
        #"$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/norm_seg/norm_k_3/bs_16_lr_1em3_NO_grad_clip",
            
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_9_norm_w_bn_layer_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_30_norm_w_bn_layer_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_42_norm_w_bn_layer_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_109_norm_w_bn_layer_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_232_norm_w_bn_layer_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_620_norm_w_bn_layer_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_937_norm_w_bn_layer_aug_on_fly",

        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_9_min_max_norm_per_img_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_30_min_max_norm_per_img_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_42_min_max_norm_per_img_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_109_min_max_norm_per_img_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_232_min_max_norm_per_img_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_620_min_max_norm_per_img_aug_on_fly",
        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_937_min_max_norm_per_img_aug_on_fly",

        # "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_aug_on_fly_num_ch_128_64",
        #"$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_aug_on_fly_num_ch_128_64_32_16",
        "$RESULTS_DIR/subcortical_structures/segmentation/abide_stanford/dino/large/bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_aug_on_fly_num_ch_128_64_32_16_upsample_type_transposed",
    ),

    # "hcp_t2": (
    #     "$RESULTS_DIR/subcortical_structures/segmentation/hcp_t2/norm_seg/norm_k_3/bs_16_lr_1em3_NO_grad_clip",
    # )
}

# Command format
# :====================================:
if slurm_jobs:
    account = "bmic"
    gpu_type = "titan_xp|geforce_rtx_2080_ti|geforce_gtx_1080_ti|titan_x"
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
