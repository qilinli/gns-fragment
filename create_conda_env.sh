#!/bin/bash
## bash commands for creating gns conda environment

conda create -y -n gns python=3.9
conda activate gns
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y pyg -c pyg
conda install -y scikit-learn-intelex
cd work/gns_1GPU/
conda install -y --file requirements.txt 
