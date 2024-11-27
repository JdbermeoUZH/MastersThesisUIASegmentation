import os

# Not frequently changed
smooth = 0
fg_only = False
epsilon = 1e-10
wandb_log = True
optimizer_type = "adamW"
learning_rate = 1e-4
warmup_steps = 0.05
weight_decay = 1e-5

# Frequently changed
account = "bmic"
epochs = 1000
max_grad_norm = 1.0

# Command format
# :====================================:
base_command = (
    f"sbatch --account='{account}' train_dino_seg_hcp.sh "
    + f"--wandb_log {wandb_log} "
    + f"--smooth {smooth} "
    + f"--fg_only {fg_only} "
    + f"--epsilon {epsilon} "
    + f"--optimizer_type {optimizer_type} "
    + f"--learning_rate {learning_rate} "
    + f"--warmup_steps {warmup_steps} "
    + f"--weight_decay {weight_decay} "
    + f"--max_grad_norm {max_grad_norm} "
    + f"--epochs {epochs} "
)

# Training combinations
# :====================================:
exps = (
    # ("wmh", "umc", "small", 48, "dice_loss_smoothing_den_1em10_opt_param_kerem"),
    # ("wmh", "umc", "base", 16, "dice_loss_smoothing_den_1em10_opt_param_kerem"),
    # ("wmh", "umc", "large", 16, 2, "dice_loss_smoothing_den_1em10_opt_param_kerem"),
    ("wmh", "nuhs", "large", 16, 2, "dice_loss_smoothing_den_1em10_opt_param_kerem"),
    ("wmh", "vu", "large", 16, 2, "dice_loss_smoothing_den_1em10_opt_param_kerem"),
    (
        "wmh",
        "umc_w_synthseg_labels",
        "large",
        16,
        2,
        "dice_loss_smoothing_den_1em10_opt_param_kerem",
    ),
    (
        "wmh",
        "nuhs_w_synthseg_labels",
        "large",
        16,
        2,
        "dice_loss_smoothing_den_1em10_opt_param_kerem",
    ),
    (
        "wmh",
        "vu_w_synthseg_labels",
        "large",
        16,
        2,
        "dice_loss_smoothing_den_1em10_opt_param_kerem",
    ),
)

for dataset_type, dataset, dino_model, batch_size, grad_acc_steps, exp_name in exps:

    logdir = os.path.join(
        os.environ["RESULTS_DIR"],
        dataset_type,
        "segmentation",
        dataset,
        "dino",
        dino_model,
        f"{exp_name}_bs_{batch_size}_grad_acc_{grad_acc_steps}_lr_{learning_rate}",
    )

    print(
        base_command
        + f"--dataset {dataset} "
        + f"--dino_model {dino_model} "
        + f"--batch_size {batch_size} "
        + f"--grad_acc_steps {grad_acc_steps} "
        + f"--logdir {logdir}"
    )

    os.system(
        base_command
        + f"--dataset {dataset} "
        + f"--dino_model {dino_model} "
        + f"--batch_size {batch_size} "
        + f"--logdir {logdir}"
    )
