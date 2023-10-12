# Conda Environment Setup

We train on Matrix without RLbench, and evaluate locally with RLBench.

Environment setup on both Matrix and locally:
```
conda create -n analogical_manipulation python=3.9
conda activate analogical_manipulation;
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia;
pip install numpy pillow einops typed-argument-parser tqdm transformers absl-py matplotlib scipy tensorboard opencv-python diffusers blosc trimesh wandb open3d;
pip install git+https://github.com/openai/CLIP.git;
```

To install RLBench locally:
```
# Install PyRep
cd PyRep; 
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz; 
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz;
echo "export COPPELIASIM_ROOT=/home/zhouxian/git/hiveformer/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> ~/.bashrc; 
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT" >> ~/.bashrc;
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> ~/.bashrc;
source ~/.bashrc;
pip install -r requirements.txt; pip install -e .; cd ..

# Install RLBench
cd RLBench; pip install -r requirements.txt; pip install -e .; cd ..;
sudo apt-get update; sudo apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev;
sudo nvidia-xconfig -a --virtual=1280x1024;
wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb --no-check-certificate;
sudo dpkg -i virtualgl*.deb; rm virtualgl*.deb;
sudo reboot  # Need to reboot for changes to take effect
```

## Dataset Generation

See `data_preprocessing` folder.

## Training Act3D

See `scripts/train_act3d.sh`.

## Training a trajectory diffusion model

See `scripts/train_trajectory.sh`.

## Evaluation

See `online_evaluation/eval.sh`.
