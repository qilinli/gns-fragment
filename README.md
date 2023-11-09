# Fragmentation Graph Network (FGN)
> This is the official implementation of FGN. 
The code is heavily based on the [Pytorch version of GNS](https://github.com/geoelements/gns) and [Tensorflow version of GNS](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate).

Qilin Li, Zitong Wang, Wensu Chen, Ling Li, Hong Hao, Curtin University

## Installation
Create python environmen. The code is tested with Python 3.11 and Pytorch 2.1.
```
conda create -n fgn python=3.11
```
Install Pytorch. Recommend using GPU version if memory > 20GB
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
Install [PyG](https://github.com/pyg-team/pytorch_geometric).
```
conda install pyg -c pyg
```
Install other dependencies.
```
pip install -r requirements.txt
```

## Inference
```
python -m gns.inference
```
