### from Euler env
# :------------------
module load stack/2024-06 python_cuda/3.11.6 2>/dev/null

python -m venv --system-site-packages tta_ddpm
source tta_ddpm/bin/activate

pip install accelerate
pip install git+https://github.com/huggingface/diffusers
pip install transformers

pip install nibabel matplotlib pandas scikit-learn tdigest scikit-image wandb kornia h5py torchmetrics

pip install denoising_diffusion_pytorch

### from scratch
# :------------------
module load stack/2024-06 python_cuda/3.11.6 2>/dev/null
python -m venv tta_ddpm

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate
pip install git+https://github.com/huggingface/diffusers
pip install transformers
pip install denoising_diffusion_pytorch

pip install nibabel matplotlib pandas scikit-learn tdigest scikit-image wandb kornia h5py torchmetrics
#pip install xformers
#pip install bitsandbytes

## with conda
# :------------------
source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh

conda create -n tta_ddpm_cuda_11 python=3.11.6 -c conda-forge -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge diffusers -y
conda install conda-forge::transformers -y
conda install -c conda-forge torchmetrics

pip install nibabel matplotlib pandas scikit-learn tdigest scikit-image wandb kornia h5py torchmetrics

#pip install xformers
#pip install bitsandbytes

# Copy envs
virtualenv-clone venv1/ venv2/