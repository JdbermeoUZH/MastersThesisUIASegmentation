import os

# Not frequently changed
# :====================================:

# Loss fn 
smooth = 0
fg_only = False
epsilon = 1e-10

# dataset
num_workers = 3
precalculated_fts = False # If False, agumentations and dino features are calculated on the fly
load_dataset_in_memory = True
node_data_path = "/scratch/${USER}/data"

# Model params
decoder_type = "ResNet"
num_channels = (128, 64)

pca_path = "${RESULTS_DIR}/subcortical_structures/pca/abide_stanford/incremental_pca/dino/large/hierachies_0_1_2_new_attempt/ipca_last.pkl"
pca_components = [9, 30, 42, 109, 232, 620, 937]

# opt params
max_grad_norm = None
optimizer_type = "adam" # "adamW"
learning_rate = 1e-3 # 2e-5
warmup_steps = 0. # 0.05
weight_decay = 0 # 1e-5

# logging
wandb_log = True

# Slurm settings
account = "bmic"
gpu_type = 'titan_xp|geforce_rtx_2080_ti|geforce_gtx_1080_ti|titan_x'
gpu_type_command = f"--constraint='{gpu_type}'" if gpu_type else ""

# join all elements in num_channels in a single string with a space
num_channels_str = ' '.join(map(str, num_channels))


# Command format
# :====================================:
base_command = (
    f"sbatch --account='{account}' {gpu_type_command} train_dino_seg.sh "
    + f"--wandb_log {wandb_log} "
    + f"--smooth {smooth} "
    + f"--fg_only {fg_only} "
    + f"--epsilon {epsilon} "
    + f"--optimizer_type {optimizer_type} "
    + f"--learning_rate {learning_rate} "
    + f"--warmup_steps {warmup_steps} "
    + f"--weight_decay {weight_decay} "
    + (f"--max_grad_norm {max_grad_norm} " if max_grad_norm else "")
    + f"--num_workers {num_workers} "
    + f"--precalculated_fts {precalculated_fts} "
    + f"--load_dataset_in_memory {load_dataset_in_memory} "
    + f"--node_data_path {node_data_path} "
    + f"--decoder_type {decoder_type} "
    + f"--num_channels {' '.join(map(str, num_channels))} "
    + f"--pca_path {pca_path} "
)

# Training combinations
# :====================================:
exps = (
    # ("wmh", "umc", "small", 48, "dice_loss_smoothing_den_1em10_opt_param_kerem"),
    # ("wmh", "umc", "base", 16, "dice_loss_smoothing_den_1em10_opt_param_kerem"),
    # ("wmh", "umc", "large", 16, 2, "dice_loss_smoothing_den_1em10_opt_param_kerem"),
    {
        "dataset_type": "subcortical_structures",
        "dataset": "abide_stanford",
        "dino_model": "large",
        "epochs": 150,
        "batch_size": 16,
        "grad_acc_steps": 1,
        "pc_norm_type": 'per_img', # 'per_img' or 'bn_layer',
        "exp_name_base":  "bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_{PCA}_min_max_norm_per_img_aug_on_fly"
    },
    {
        "dataset_type": "subcortical_structures",
        "dataset": "abide_stanford",
        "dino_model": "large",
        "epochs": 150,
        "batch_size": 16,
        "grad_acc_steps": 1,
        "pc_norm_type": 'bn_layer', # 'per_img' or 'bn_layer',
        "exp_name_base":  "bs_16_lr_1em3_NO_grad_clip_NO_weight_decay_hier_2_PCA_num_PCA_{PCA}_norm_w_bn_layer_aug_on_fly"
    },
)

for exp_dict in exps:
    dataset_type = exp_dict["dataset_type"]
    dataset = exp_dict["dataset"]
    dino_model = exp_dict["dino_model"]
    epochs = exp_dict["epochs"]
    batch_size = exp_dict["batch_size"]
    grad_acc_steps = exp_dict["grad_acc_steps"]
    exp_name_base = exp_dict["exp_name_base"]
    pc_norm_type = exp_dict["pc_norm_type"]

    base_command += (
        f"--dataset {dataset} "
        + f"--dino_model {dino_model} "
        + f"--epochs {epochs} "
        + f"--batch_size {batch_size} "
        + f"--grad_acc_steps {grad_acc_steps} "
        + f"--pc_norm_type {pc_norm_type} "
    )

    for n_pcs in pca_components:
        exp_name = exp_name_base.format(PCA=n_pcs)
            

        logdir = os.path.join(
            os.environ["RESULTS_DIR"],
            dataset_type,
            "segmentation",
            dataset,
            "dino",
            dino_model,
            f"{exp_name}",
        )
        
        wandb_run_name = os.path.join(
            dataset,
            "dino",
            dino_model,
            decoder_type.lower(),
            exp_name
        )

        command = base_command + (            
            f"--wandb_run_name {wandb_run_name} "
            + f"--logdir {logdir} "
            + f"--num_pca_components {n_pcs} "
        )

        print(command)

        os.system(command)
