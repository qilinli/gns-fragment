# This code is to test the validity of data reading
# It generates an animation, where the 1st frame is color coded by particle types
# Subsequent frames are color coded by particle strains

import numpy as np


def check_data_integrity(particle_trajectories, particle_strains, particle_type):
    error_messages = []
    nstep, nparticles, dim = particle_trajectories.shape
    
    # Shape checks (example)
    if dim != 3:
        error_messages.append(f"Particle trajectories shape mismatch: expected 3 in the last dimension, got {dim}")
    if particle_type.shape != (nparticles,):
        error_messages.append(f"Particle_type shape mismatch: expected ({nparticles},), got {particle_type.shape}")
    if particle_strains.shape != (nstep, nparticles):
        error_messages.append(f"Particle_strains shape mismatch: expected ({nstep}, {nparticles}), got {particle_strains.shape}")
        
    # Missing value check
    if np.isnan(particle_trajectories).any():
        error_messages.append("Particle trajectories contain NaN values")
    if np.isnan(particle_strains).any():
        error_messages.append("Particle strains contain NaN values")
    
    # Value Range CHecks:
    x_min, x_max = particle_trajectories[0,:,0].min(), particle_trajectories[0,:,0].max()
    y_min, y_max = particle_trajectories[0,:,1].min(), particle_trajectories[0,:,1].max()
    z_min, z_max = particle_trajectories[0,:,2].min(), particle_trajectories[0,:,2].max()

    # Check if the values are within the specified ranges and collect error messages
    if not (-500 <= x_min <= 500 and -500 <= x_max <= 500):
        error_messages.append(f"X coordinates out of bounds: min {x_min}, max {x_max}")
    if not (-1000 <= y_min <= 1000 and -1000 <= y_max <= 1000):
        error_messages.append(f"Y coordinates out of bounds: min {y_min}, max {y_max}")
    if not (0 <= z_min <= 200 and 0 <= z_max <= 200):
        error_messages.append(f"Z coordinates out of bounds: min {z_min}, max {z_max}")

    # Raise a single ValueError if there are any issues
    if error_messages:
        raise ValueError("Data integrity check failed with the following issues:\n" + "\n".join(error_messages))
        
        
if __name__ == "__main__":
    pass