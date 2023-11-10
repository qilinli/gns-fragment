# This code is to read all d3plots in a root folder and
# Save them npz files for future usage

from pathlib import Path
import tqdm
import numpy as np
from read_d3plot import extract_trajectory_type_strain, D3plot
from check_data_integrity import check_data_integrity
from data_processing import enforce_eps_non_decreasing, timestep_downsample

# Define the root folder as a Path object
root_dir = Path(r'C:\Users\kylin\OneDrive - Curtin\research\civil_engineering\data\FGN')

# Use glob to find all d3plot files; this returns a generator
all_d3plots = root_dir.rglob('d3plot')

# Create output directory

# Use tqdm for progress indication
successful_read_count = 0
for path_to_d3plot in tqdm.tqdm(all_d3plots):
    d3plot = D3plot(str(path_to_d3plot))

    # Read data from d3plot
    particle_trajectories, particle_type, particle_strains = extract_trajectory_type_strain(d3plot)
    
    # Data processing
    particle_strains = enforce_eps_non_decreasing(particle_strains)
    particle_trajectories, particle_strains = timestep_downsample(particle_trajectories, particle_strains)
    
    # Perform integrity check
    try:
        check_data_integrity(particle_trajectories, particle_strains, particle_type)
    except ValueError as e:
        # Handle the error as you see fit (print, log, skip, etc.)
        print(f"Data integrity check failed for {path_to_d3plot}: {e}")
        print(f"Skip {path_to_d3plot}.")   # Skip this file or use other error handling
        continue
    
    # Create a path for the npz file in the same directory as the d3plot file
    path_to_npz = path_to_d3plot.parent / (path_to_d3plot.parent.name + '.npz')

    # Save data as an npz file
    with path_to_npz.open('wb') as f:
        np.savez(f, 
                 particle_trajectories=particle_trajectories, 
                 particle_strains=particle_strains, 
                 particle_type=particle_type
                 )
    print(f"Successfully saved: {path_to_npz}")
    successful_read_count += 1
print("==============Finished data reading.================")    
print(f"Successfully read and saved {successful_read_count} cases.")
