# Minimum to run nnUNet
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install nnUnet
pip install nnunetv2
#git clone https://github.com/MIC-DKFZ/nnUNet.git
#cd nnUNet
#pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
conda install graphviz
pip install IPython

# Extra stuff
pip install jupyter