import numpy as np
import glob
import json
import random
import math
import pathlib

from parse_lsdyna_simulation import parse_simulation, parse_simulation_strain         


dataset = 'Concrete2D-I-Step25'
in_dir = f'/home/jovyan/share/gns_data/Concrete2D-DYNA/'
out_dir = f'/home/jovyan/share/gns_data/{dataset}/'
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

# Grab all simulation cases from corresponding data folder
simulations = glob.glob(in_dir + 'coordinates_eps/*.txt')
random.shuffle(simulations)

## Larger step size leads to shorter trajectory and hence better rollout performance
## But lower precision of the simulation
## Current simulation are of absolute time 0.5 seconds
## Step size=1 means 500 steps, each of which 1 ms
## step size=5 means 100 steps, each of which 5 ms
STEP_SIZE = 25

## For the pnormalisation of data
# strain_C_mean = 1.138
# strain_C_std = 0.831
# strain_I_mean = 1.503
# strain_I_std = 0.735
MAX_I = np.array([325, 150])
MIN_I = np.array([-325, -30])
MAX_C = np.array([325, 95])
MIN_C = np.array([-325, -15])
strain_min, strain_max = 0, 2

## Fixed hold-out validation and testing, for reproducibility
## GNN performance seems better with cases of larger loading (higher contact velocity) 
I_valid_trajectory = ['B_80_320_Aa_120', 'B_80_480_Ac_120', 'R_80_320_Aa_80', 'R_80_480_Ab_160', 
                      'R_80_640_Ac_160', 'S_80_320_Aa_160', 'S_80_480_Ab_80', 'S_80_640_Ac_120']

I_test_trajectory = ['B_80_320_Aa_80', 'B_80_480_Ac_160', 'R_80_320_Aa_160', 'R_80_480_Ab_120', 
                     'R_80_640_Ac_120', 'S_80_320_Aa_120', 'S_80_480_Ab_160', 'S_80_640_Ac_160']

C_valid_trajectory = ['C_80_320_Aa_12', 'C_80_320_Bc_8', 'C_80_320_Ca_16', 'C_80_320_Cb_20', 'C_80_480_Ab_20', 
                      'C_80_480_Bb_12', 'C_80_480_Cc_20', 'C_80_480_Ca_16', 'C_80_640_Aa_16', 'C_80_640_Bb_12']

C_test_trajectory = ['C_80_320_Aa_16', 'C_80_320_Bc_12', 'C_80_320_Ca_20', 'C_80_320_Cb_16', 'C_80_480_Ab_16', 
                     'C_80_480_Bb_8', 'C_80_480_Cc_12', 'C_80_480_Ca_20', 'C_80_640_Aa_12', 'C_80_640_Bb_20']

## Different datasets have different parameters
if 'Concrete2D-I' in dataset:
    simulations = [f for f in simulations if "C_" not in f and "40." not in f]
    valid_trajectory = I_valid_trajectory
    test_trajectory = I_test_trajectory
    MAX = MAX_I
    MIN = MIN_I
elif 'Concrete2D-C' in dataset:
    simulations = [f for f in simulations if "C_" in f]
    valid_trajectory = C_valid_trajectory
    test_trajectory = C_test_trajectory
    MAX = MAX_C
    MIN = MIN_C
else:
    print("Dataset not found!")

## Initialisation placeholders for data
n_trajectory = len(simulations)
ds_train, ds_valid, ds_test = {}, {}, {}
vels = np.array([]).reshape(0, 2)
accs = np.array([]).reshape(0, 2)
file_train, file_valid, file_test = [], [], []

## Main loop for data extraction
for idx, simulation in enumerate(simulations):
    print(f"{idx}/{n_trajectory} Reading {simulation}...")
    positions, particle_types, eps = parse_simulation(simulation)
    
    # For constant velocity, get rid of the last 4 boundary particles
    traj_name = simulation.split('/')[-1][:-4]
    if traj_name.startswith('C'):
        positions = positions[::1,:-4:1,::1]
        particle_types = particle_types[:-4]
        eps = eps[::1, :-4:1]
        
    ## Preprocessing position
    # Normalisation to [0, 1]
    positions = positions[::STEP_SIZE,::1,::1]    
    positions = (positions - MIN) / (MAX - MIN) 
    
    # After normalisation the y-axis of beam is stretched
    # Apply a scalling factor to retain x/y ratio
    # so that the shape of beam does not change
    # it also means the connection radius will work similarlly for both x and y
    y_scaling_factor = (MAX - MIN)[0] / (MAX - MIN)[1]
    positions[:,:,1] = positions[:,:,1] / y_scaling_factor  
    
    # Preprocessing eps
    # Currently [0,1] normalisation in use
    # This is actually unnecessary as the raw range is [0,2]
    # Scale it down to [0, 1] effectively reduces the contribution of eps in loss function
    # TODO: Probbably can leave it raw in the future
    eps = eps[::STEP_SIZE, ::1]
    eps[eps < 1e-8] = 0
    # eps = (eps - strain_min) / (strain_max - strain_min)
    ## Standardisation maeks more sense but empirically not so
    # strains = (strains - strain_mean) / strain_std
    
    # Preprocessing mps
    # raw mps contains many small value like 1e-8, 1e-39 and negative 1e-7
    # while negative value has physical meaning, it is too small
    # No normalisation is performed as empirically reduces performance
    if 'mps' in dataset:
        mps = parse_simulation_strain(simulation)
        if traj_name.startswith('C'):
            mps = mps[::1, :-4:1]
        mps = mps[::STEP_SIZE, ::1]
        mps[mps < 1e-8] = 0
    
    strains = mps if 'mps' in dataset else eps
    
    # print for debug
    print(f"Position min:{positions.min(axis=(0,1))}, max:{positions.max(axis=(0,1))}")
    print(f"Strain min:{strains.min(axis=(0,1))}, max:{strains.max(axis=(0,1))}")
    print(f"Shape, pos: {positions.shape}, types: {particle_types.shape}, strain: {strains.shape}")
    print(f"Unique particle types: {np.unique(particle_types)}")
    
    # Data splits: train(80%), valid(10%), test(10%)
    key = 'trajectory_' + str(idx)
    if traj_name in valid_trajectory:
        print('to valid')
        ds_valid[key] = [positions, particle_types, strains]
        file_valid.append(traj_name)
    elif traj_name in test_trajectory:
        print('to test')
        ds_test[key] = [positions, particle_types, strains]
        file_test.append(traj_name)
    else:
        print('to train')
        ds_train[key] = [positions, particle_types, strains]
        file_train.append(traj_name)
        
    # Extract Vel and Acc statistics
    # positions of shape [timestep, particles, dimensions]
    vel_trajectory = positions[1:,:,:] - positions[:-1,:,:]
    acc_trajectory = vel_trajectory[1:,:,:]- vel_trajectory[:-1,:,:]
    
    vels = np.concatenate((vels, vel_trajectory.reshape(-1, 2)), axis=0)
    accs = np.concatenate((accs, acc_trajectory.reshape(-1, 2)), axis=0)

# Extract vel, acc statistics for normalisation
vel_mean, vel_std = list(vels.mean(axis=0)), list(vels.std(axis=0))
acc_mean, acc_std = list(accs.mean(axis=0)), list(accs.std(axis=0))

# Save datasets in numpy format
np.savez(out_dir + 'train.npz', **ds_train)
np.savez(out_dir + 'valid.npz', **ds_valid)
np.savez(out_dir + 'test.npz', **ds_test)

print(f"{len(ds_train)} trajectories saved to train.npz.")
print(f"{len(ds_valid)} trajectories saved to valid.npz.")
print(f"{len(ds_test)}  trajectories saved to test.npz.")

# Save meta data
in_file = '/home/jovyan/share/gns_data/WaterDropSample/metadata.json'
out_file = f'/home/jovyan/share/gns_data/{dataset}/metadata.json'

with open(in_file, 'r') as f:
    meta_data = json.load(f)

# meta_data['bounds'] = [[-200, 200], [0, 100]]
# The origin of simulation domain is at bottom center, and x in [-165, 165], y in [-10, 85].
# Particle radius r is 1.25 mm, and the connection Radius R is around 6r to 7r, or [7.5, 8.75] (24 neighbors, maybe more)
# In GNN, the suggested connection radius is 4.5r, or 5.625 mm (aounrd 20 neighbors)
# If R is 6mm before normalization, then it is 0.016 where 6/370 = x/1

# 0.04 (normalized) or 7.5 (unnormalized) for around 24 neighbours
# Note this parameter has been surpassed by another one in train.py
meta_data['default_connectivity_radius'] = 0.01 
meta_data['sequence_length'] = positions.shape[0]
meta_data['vel_mean'] = vel_mean
meta_data['vel_std'] = vel_std
meta_data['acc_mean'] = acc_mean
meta_data['acc_std'] = acc_std
# meta_data['strain_mean'] = strain_mean
# meta_data['strain_std'] = strain_std

meta_data['dt'] = 0.001 * STEP_SIZE
if 'Concrete2D-C' in dataset:
    meta_data['bounds'] = [[0,1], [0, 0.17]]
else:
    meta_data['bounds'] = [[0,1], [0, 0.25]]
meta_data['file_train'] = file_train
meta_data['file_valid'] = file_valid
meta_data['file_test'] = file_test
print(meta_data)

with open(out_file, 'w') as f:
    json.dump(meta_data, f)