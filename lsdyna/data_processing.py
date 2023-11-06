# This code is to test the validity of data reading
# It generates an animation, where the 1st frame is color coded by particle types
# Subsequent frames are color coded by particle strains

import numpy as np

INPUT_SEQUENCE_LENGTH = 10
STEP_SIZE = 6

def enforce_eps_non_decreasing(particle_strains):
    # Compute the differences between adjacent time steps
    strains_diff = np.diff(particle_strains, axis=0)

    # Set any negative differences to zero
    strains_diff[strains_diff < 0] = 0

    # Reconstruct the corrected strains using cumulative sum,
    # starting with the initial strain values
    corrected_strains = np.concatenate((particle_strains[:1, :], strains_diff), axis=0).cumsum(axis=0)
    return corrected_strains


def timestep_downsample(particle_trajectories, particle_strains):
    particle_trajectories_downsampled = particle_trajectories[::STEP_SIZE]
    particle_trajectories_downsampled = particle_trajectories_downsampled[-INPUT_SEQUENCE_LENGTH:]
    
    particle_strains_downsampled = particle_strains[::STEP_SIZE]
    # Convert EPS to ResEPS (residual eps)
    strains_diff = np.diff(particle_strains_downsampled, axis=0)
    ## The initial strain should be the strain at step=-10 after downsampling
    init_strains = particle_strains_downsampled[-INPUT_SEQUENCE_LENGTH:-INPUT_SEQUENCE_LENGTH+1, :]
    ## The final ResEPS should combine the init_strain and following 9 steps of strain_dff
    particle_strains_downsampled = np.concatenate((init_strains, strains_diff[-INPUT_SEQUENCE_LENGTH+1:]), axis=0)
    
    return particle_trajectories_downsampled, particle_strains_downsampled

