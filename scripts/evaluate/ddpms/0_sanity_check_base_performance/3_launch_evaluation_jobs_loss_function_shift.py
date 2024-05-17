import os
import argparse


#-------------------------------------
# Default args
output_dir_def = '/scratch_net/biwidl319/jbermeo/results/brain/ddpm/'
num_iterations_def = 100 
num_t_samples_per_img_def = 20 
batch_size_def = 16
#-------------------------------------



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_sd', type=str)
    parser.add_argument('dataset_name_td', type=str)
    parser.add_argument('ddpm_dir', type=str)
    parser.add_argument('cpt_number', type=int)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=output_dir_def)
    parser.add_argument('--num_iterations', type=int, default=num_iterations_def)
    parser.add_argument('--num_t_samples_per_img', type=int, default=num_t_samples_per_img_def)
    parser.add_argument('--batch_size', type=int, default=batch_size_def)
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    
    exp_name = args.exp_name if args.exp_name else os.path.basename(args.ddpm_dir)
    
    domain_combinations = {
        'sd_sd': [args.dataset_name_sd, args.dataset_name_sd],
        'sd_td': [args.dataset_name_sd, args.dataset_name_td],
    }
    
    split_combinations = {
        'train_train': ['train', 'train'],
        'train_val': ['train', 'val'],
        'val_val': ['val', 'val'],  
    }
    
    mismatches = ['none', 'same_patient_similar_labels', 'same_patient_very_different_labels']
    
    for dc_name, (sd_name, td_name) in domain_combinations.items():
        print(f'Running experiments for domain combination: {dc_name}')
        for split_comb_name, (sd_split, td_split) in split_combinations.items():
            for mismatch in mismatches:
                if dc_name == 'sd_sd' and mismatch == 'none' and split_comb_name != 'train_val':
                    continue
                
                if dc_name == 'sd_sd' and mismatch != 'none' and split_comb_name == 'train_val':
                    continue
                
                if dc_name == 'sd_td' and not (split_comb_name == 'train_train' and mismatch == 'none'):
                    continue
                
                print(f'Running experiments for split combination: {split_comb_name}')
                print(f'Running experiments for mismatch mode: {mismatch}')
            
                command = f"sbatch 3_ddpm_loss_multiple_t_and_imgs.sh" + \
                          f" --ddpm_dir {args.ddpm_dir}" + \
                          f" --cpt_fn model-{args.cpt_number}.pt" + \
                          f" --num_iterations {args.num_iterations}" + \
                          f" --num_t_samples_per_img {args.num_t_samples_per_img}" + \
                          f" --batch_size {args.batch_size}" + \
                          f" --dataset_sd {sd_name} --split_sd {sd_split}" + \
                          f" --dataset_td {td_name} --split_td {td_split}" + \
                          f" --out_dir {args.output_dir}" + \
                          f" --mismatch_mode {mismatch}" + \
                          f" --exp_name {exp_name}" 

                print('Running command: \n', command, '\n\n', '-'*50, '\n\n')
                os.system(command)
        print('#' * 100 + '\n\n')