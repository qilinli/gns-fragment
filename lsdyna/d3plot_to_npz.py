# This code is to read all d3plots in a root folder and
# Save them npz files for future usage

from pathlib import Path
import tqdm
import numpy as np
from read_d3plot import extract_trajectory_type_strain, enforce_eps_non_decreasing, D3plot
from check_data_integrity import check_data_integrity

# Define the root folder as a Path object
root_folder = Path(r'C:\Users\kylin\OneDrive - Curtin\research\civil_engineering\data\FGN\C30_120_6_0.4')

# Use glob to find all d3plot files; this returns a generator
all_d3plots = root_folder.rglob('d3plot')

# Use tqdm for progress indication
for path_to_d3plot in tqdm.tqdm(all_d3plots):
    d3plot = D3plot(str(path_to_d3plot))

    # Read data using your custom functions
    particle_trajectories, particle_type, particle_strains = extract_trajectory_type_strain(d3plot)
    particle_strains = enforce_eps_non_decreasing(particle_strains)
    
    # Check data integrity
    check_data_integrity(particle_trajectories, particle_type, particle_strains)
    
    # Create a path for the npz file in the same directory as the d3plot file
    new_name = f"{path_to_d3plot.stem}6.npz"
    path_to_npz = path_to_d3plot.with_name(new_name)

    # Save data as an npz file
    with path_to_npz.open('wb') as f:
        np.savez(f, 
                 particle_trajectories=particle_trajectories, 
                 particle_strains=particle_strains, 
                 particle_type=particle_type
                 )
        
