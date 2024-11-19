import os


# Params for each job
# :====================================:
num_workers = 2
hierarchy_levels = 2
num_aug_epochs = 5
gpu_type = None #"geforce_gtx_titan_x"

# Command format
# :====================================:
account = "bmic"
base_command = f"sbatch --account={account}" 
base_command += f" --constraint='{gpu_type}'" if gpu_type is not None else " "
base_command += " precompute_dino_features.sh"

base_command += (
    f" --num_workers {num_workers}"
    f" --hierarchy_levels {hierarchy_levels}"
    f" --num_epochs {num_aug_epochs}"
)

# Specific run param combination
# :====================================:
dino_model_sizes = {
    "base": {"batch_size": 32},
    "large": {"batch_size": 16},
    "giant": {"batch_size": 8},
}
datasets = [
    # "hcp_t1",
    # "hcp_t2",
    # "abide_caltech",
    # "abide_stanford",
    #"umc",
    # "umc_w_synthseg_labels",
    "nuhs",
    # "nuhs_w_synthseg_labels",
    #"vu",
    # "vu_w_synthseg_labels",
]

splits = ["train", "val", "test"]

# Run the commands
# :====================================:

for dino_model_size, params in dino_model_sizes.items():
    print(f"Precomuping features with Dino: {dino_model_size}")

    for dataset in datasets:
        for split in splits:
            print(f"Launching job for {dataset} - {split} - {dino_model_size}")
            slurm_command = base_command + (
                f" --dino_model {dino_model_size}"
                f" --dataset {dataset}"
                f" --batch_size {params['batch_size']}"
                f" --splits {split}"
            )

            os.system(slurm_command)
