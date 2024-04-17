import os
import argparse

#---------- Default args
synthseg_dir_def            = '/scratch_net/biwidl319/jbermeo/SynthSeg/'
input_dir_def               = '/scratch_net/biwidl319/jbermeo/data/wmh_miccai'
output_dir_def              = '/scratch_net/biwidl319/jbermeo/data/preprocessed/synthseg_predictions/on_peprocessed_vols/wmh'
#----------


parse_boolean = lambda x: x.lower() == 'true'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluates the SynthSeg model')
    parser.add_argument('dataset', type=str, help='Dataset to use')
    parser.add_argument('--synthseg_dir', type=str, default=synthseg_dir_def, help='Path to the SynthSeg script')
    parser.add_argument('--input_dir', type=str, default=input_dir_def, help='Path to save the nifti files')
    parser.add_argument('--output_dir', type=str, default=output_dir_def, help='Path to save the nifti files')
    
    args = parser.parse_args()
    
    splits = ['train', 'val', 'test']
    script_path = './scripts/commands/SynthSeg_predict.py'
    
    os.chdir(args.synthseg_dir)
    
    print('Dataset:', args.dataset)
    
    for split in splits:
        print('Split:', split)
        input_dir = os.path.join(args.input_dir, args.dataset, split, 'imgs')
        output_dir = os.path.join(args.output_dir, args.dataset, split)
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(input_dir):
            print(f'WARNING: {input_dir} does not exist')
            continue
        
        try:
            os.system(
                f'python {script_path}' +
                f' --i {input_dir} --o {output_dir}' +
                f' --resample {output_dir}' +
                ' --robust')
        except Exception as e:
            print(f'ERROR: {e}')
            continue
    
    print('Done')