# This code is to extract data from d3plot, generated by LS-DYNA
# based on lasso-python: https://github.com/open-lasso-python/lasso-python
# Currently it reads trajectory, strains, and types of each element(particle)


import numpy as np
from lasso.dyna import D3plot, ArrayType, FilterType

def extract_trajectory_type_strain(d3plot):
    """Read particle (element) trajectory of coordinates.
    
    Input: the d3plot data
    Output: particle_trajectories of shape [ntimesteps, nparticles, 3]
    """
    
    ## node_displacement is actually node coords in all steps, shape [nstep, nnodes, 3]
    node_trajectories = d3plot.arrays["node_displacement"]
    ## Each solid element (cubic) is defined by 8 nodes
    # element_solid_node_indexes = d3plot.arrays["element_solid_node_indexes"]
    ## each beam involves 2 nodes, but the array shows 5 with 3rd being the same as 2nd
    ## and 4th, 5th looks unrelated
    ## Using LS-PrePost outputs 3 nodes per beam, with 3rd also being the same as 2nd
    ## Therefore, only the first 2 nodes are used
    element_beam_node_indexes = d3plot.arrays["element_beam_node_indexes"][:, :2]
    
    # Convert the solid node indexes to a set for quick look-up
    sph_node_indexes = d3plot.arrays["sph_node_indexes"]
    SPH_trajectories = node_trajectories[:, sph_node_indexes, :]

    element_beam_node_indexes = np.unique(element_beam_node_indexes)
    
    element_beam_trajectories = node_trajectories[:, element_beam_node_indexes, :]

    particle_trajectories = np.concatenate((SPH_trajectories, element_beam_trajectories), axis=1)

    # Derive particle types, 0 concrete, 1 rebar, 2 boundary
    # boundary is always 150 mm on the two ends of y-axis
    SPH_types = np.zeros(SPH_trajectories.shape[1])
    beam_types = np.ones(element_beam_trajectories.shape[1])
    particle_type = np.concatenate((SPH_types, beam_types), axis=0)
    LEFT_BOUNDARY = -855    # particle_trajectories[0, :, 1].min() + 150, this not aligned with the data from txt
    RIGHT_BOUNDARY = 855    # particle_trajectories[0, :, 1].max() - 150
    mask = (particle_trajectories[0, :, 1] >= RIGHT_BOUNDARY) | (particle_trajectories[0, :, 1] <= LEFT_BOUNDARY)
    particle_type[mask] = 2
    
    # Strain
    solid_eps = d3plot.arrays["element_solid_effective_plastic_strain"][:, :, 0]
    beam_eps = np.zeros((solid_eps.shape[0], element_beam_trajectories.shape[1]))
    particle_strains = np.concatenate((solid_eps, beam_eps), axis=1)
    
    return particle_trajectories, particle_type, particle_strains


if __name__ == "__main__":
    path_to_d3plot = r'C:\Users\272766h\Curtin University of Technology Australia\Zitong Wang - Data generation\C30_120mm_0.4m\5kg\d3plot'
    d3plot = D3plot(path_to_d3plot)
    
    particle_trajectories, particle_type, particle_strains = extract_trajectory_type_strain(d3plot)

    print(f'Particle_trajectories of shape: {particle_trajectories.shape}')
    print(f'particle_strains of shape: {particle_strains.shape}')
    print(f'particle_types of shape: {particle_type.shape}')
    print('=== Reading complete ===')

        
    