# Minimum to run nnUNet
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install nnunetv2 
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
conda install graphviz -y
pip install IPython

# Extra stuff
pip install jupyterpip install diffusers["torch"] transformers