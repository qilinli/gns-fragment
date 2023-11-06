# This code is to test the validity of data reading
# It generates an animation, where the 1st frame is color coded by particle types
# Subsequent frames are color coded by particle strains

import numpy as np


def check_data_integrity(particle_trajectories, particle_strains, particle_type):
    nstep, nparticles, _ = particle_trajectories.shape
    
    # Shape check:
    assert particle_type.shape == (nparticles,), "Mismatch in shapes: particle_type"
    assert particle_strains.shape == (nstep, nparticles), "Mismatch in shapes: particle_strains"
    
    # Missing value check
    for array, name in zip([particle_trajectories, particle_type, particle_strains],
                           ["particle_trajectories", "particle_type", "particle_strains"]):
        if np.any(np.isnan(array)):
            print(f"Missing values detected in {name}")
    
    # Value Range CHecks:
    for i, axis in enumerate(["x", "y", "z"]):
        print(f"{axis}-axis min: {np.min(particle_trajectories[:, :, i]):.2f}, max: {np.max(particle_trajectories[:, :, i]):.2f}")
    
    if np.any((particle_strains < 0) | (particle_strains > 2)):
        print("Effective Plastic strain values out of range [0, 2]")
        
        
if __name__ == "__main__":
    pass