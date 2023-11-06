# This code is to read all d3plots in a root folder and
# Save them npz files for future usage

from pathlib import Path
import tqdm
import numpy as np
from read_d3plot import extract_trajectory_type_strain, D3plot
from check_data_integrity import check_data_integrity
from data_processing import enforce_eps_non_decreasing, timestep_downsample

# Define the root folder as a Path object
root_folder = Path(r'C:\Users\272766h\Curtin University of Technology Australia\Zitong Wang - Data generation\C30_120mm\0.8_10')

# Use glob to find all d3plot files; this returns a generator
all_d3plots = root_folder.rglob('d3plot')

# Use tqdm for progress indication
for path_to_d3plot in tqdm.tqdm(all_d3plots):
    d3plot = D3plot(str(path_to_d3plot))

    # Read data from d3plot
    particle_trajectories, particle_type, particle_strains = extract_trajectory_type_strain(d3plot)
    
    # Data processing
    particle_strains = enforce_eps_non_decreasing(particle_strains)
    particle_trajectories, particle_strains = timestep_downsample(particle_trajectories, particle_strains)
    
    # Check data integrity
    check_data_integrity(particle_trajectories, particle_strains, particle_type)
    
    # Create a path for the npz file in the same directory as the d3plot file
    new_name = f"{path_to_d3plot.stem}_0.8.npz"
    path_to_npz = path_to_d3plot.with_name(new_name)

    # Save data as an npz file
    with path_to_npz.open('wb') as f:
        np.savez(f, 
                 particle_trajectories=particle_trajectories, 
                 particle_strains=particle_strains, 
                 particle_type=particle_type
                 )
        
