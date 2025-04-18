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
cd gns-fragment
pip install -r requirements.txt
```

## Data
Data folder following structure
```
├──($data_path)/
│ ├── 0.2_4/
│   ├── d3plot
│   ├── d3plot01
│   ├── d3plot02
│   └── ...
│ ├── 0.3_4/
│ ├── 0.4_4/
│ ├── ...
│ └── meatadata.json

```
A data sample is here [OneDrive](https://curtin-my.sharepoint.com/:f:/g/personal/272766h_curtin_edu_au/Ep0xeC_hi91GvSs_0ABY9m8BcgCLgCWhn4mwKwz9AhNEZg?e=nxsAmb)

Run the following to extract all data from d3plot to numpy. Each case will result in a .npz file saved in the case subfolder, e.g., 0.2_4/0.2_4.npz, which will then be used for FGN inference.
```
python .\lsdyna\d3plot_to_npz.py --data_path=$PATH_TO_DATA
```

## Inference
FGN inference on all cases (.npz) with in the data folder including subfolder. The inference can go as many steps as you want, where each step is of 0.06 ms. Detonation_xy specifies the xy coordinates of the explosive source (z is controled by standoff distance).
```
python -m gns.inference --data_path=$PATH_TO_DATA --nsteps=81 --detonation_xy=0,0 --expanded_search=False
```

## Result
After running it should generate all figures and csvs, which are all saved in the same output folder. It should look like
```
├──($data_path)/
│ ├── 0.2_4/
|   ├── d3plot.npz
│   ├── d3plot
│   ├── d3plot01
│   ├── d3plot02
│   └── ...
│ ├── 0.3_4/
│ ├── 0.4_4/
│ ├── ...
│ ├── meatadata.json
| └── output/
|       ├── 0.2_4/
|           ├── eps/
|           ├── fragment/
|           ├── mass/
|           └── property/
|       ├── 0.3_4/
|       └── ...
```
![Example Image of EPS](/figures/eps_top_step_80.png "Effective Plastic Strain")
![Example Image of EPS](/figures/fragment_step_80.png "Effective Plastic Strain")

