import re
import numpy as np
import random

    
def parse_simulation(file):
    '''
    Extract info from LSDYNA txt, including particle coordinates, particle types, and effective plastic strain (eps)."
    Input: Txt from LYDYNA, e.g., C_80_480_Cc_20_strain.txt
    Output: np arrays of shapes, 
            tracjectory (timesteps, num_particles, 2), particle_type (num_particles,), eps (timesteps, num_particles).
    '''
    
    ## Concrete-2D-CI particle types based on particle index
    PARTICLE_TYPES ={
        'B_80_320': {'concrete': (1, 4096), 'kinematic': (4153, 4264), 'support': (4265, 4328)},
        'B_80_480': {'concrete': (1, 6144), 'kinematic': (6181, 6292), 'support': (6293, 6356)},
        'B_80_640': {'concrete': (1, 8192), 'kinematic': (8229, 8340), 'support': (8341, 8404)},
        'R_80_320': {'concrete': (1, 4096), 'kinematic': (4153, 4264), 'support': (4265, 4328)},
        'R_80_480': {'concrete': (1, 6144), 'kinematic': (6181, 6292), 'support': (6293, 6356)},
        'R_80_640': {'concrete': (1, 8192), 'kinematic': (8229, 8340), 'support': (8341, 8404)},
        'S_80_320': {'concrete': (1, 4096), 'kinematic': (4097, 4208), 'support': (4209, 4272)},
        'S_80_480': {'concrete': (1, 6144), 'kinematic': (6145, 6256), 'support': (6257, 6320)},
        'S_80_640': {'concrete': (1, 8192), 'kinematic': (8193, 8304), 'support': (8305, 8368)},
        'C_80_320': {'concrete': (1, 4096), 'kinematic': (4097, 4128), 'support': (4129, 4196)},   # For C_, the last 4 partciles are boundary
        'C_80_480': {'concrete': (1, 6144), 'kinematic': (6145, 6176), 'support': (6177, 6244)},
        'C_80_640': {'concrete': (1, 8192), 'kinematic': (8193, 8224), 'support': (8225, 8292)},
        # 2D generalize test
        'C_60_240': {'concrete': (1, 2304), 'kinematic': (8193, 8224), 'support': (8225, 8292)},   
        'C_80_560': {'concrete': (1, 7168), 'kinematic': (8193, 8224), 'support': (8225, 8292)}, 
        'S_80_400': {'concrete': (177, 5296), 'kinematic': (1, 112), 'support': (113, 176)},  
        'S_100_80': {'concrete': (177, 12976), 'kinematic': (1, 112), 'support': (113, 176)},     #suppose to be S_100_800, simplified for easy readbility       
    } 

    with open(file, 'r') as f:
        lines = f.readlines()

    # Find all "particle position" lines and "plastic strain" lines using key words
    pos_lines_start, pos_lines_end = [], []
    strain_lines_start, strain_lines_end = [], []
    for idx, line in enumerate(lines):
        if line.startswith("*NODE"):
            pos_lines_start.append(idx)
        elif line.startswith("$NODAL_RESULTS"):  # $NODAL_RESULTS,(1d) *INITIAL_VELOCITY_NODE(2d)
            pos_lines_end.append(idx)
        elif line.startswith("$RESULT OF Effective Plastic Strain"):
            strain_lines_start.append(idx)
        elif line.startswith("*END"):  
            strain_lines_end.append(idx)
            
    # Extact particle positions 
    trajectory = []
    for line_start, line_end in zip(pos_lines_start, pos_lines_end):
        pos_lines = lines[line_start+1:line_end]   # lines that contains positions in one time step
        timestep = []
        for line in pos_lines:
            num_str = re.findall(r'[-\d\.e+]+', line)  # Regular expression findign scitific numbers
            (x, y) = (float(num_str[1]), float(num_str[2]))
            timestep.append((x,y))
        trajectory.append(timestep) 
    
    # Extact particle types
    particle_types = []
    traj_name = file.split('/')[-1][:8]
    pos_lines = lines[pos_lines_start[0]+1:pos_lines_end[0]]
    for line in pos_lines:
        num_str = re.findall(r'[-\d\.e+]+', line)
        particle_id = int(num_str[0])
        if particle_id >= PARTICLE_TYPES[traj_name]['concrete'][0] and particle_id <= PARTICLE_TYPES[traj_name]['concrete'][1]:
            particle_types.append(0)   # concrete particles
        elif particle_id >= PARTICLE_TYPES[traj_name]['support'][0] and particle_id <= PARTICLE_TYPES[traj_name]['support'][1]:
            particle_types.append(1)   # boundary particles (rigid)
        elif particle_id >= PARTICLE_TYPES[traj_name]['kinematic'][0] and particle_id <= PARTICLE_TYPES[traj_name]['kinematic'][1]:
            if traj_name.startswith('C'):
                particle_types.append(3)   # kinematic particles only presents in cases with constand loading
            else:
                particle_types.append(2)   # steel particles used for cases of impact loading
        else:
            raise ValueError('particle id not defined')
    
    # Extract effective plastic strain (eps)
    strains = []
    for line_start, line_end in zip(strain_lines_start, strain_lines_end):
        strain_lines = lines[line_start+1:line_end]   # lines that contains positions in one time step
        strains_one_step = []
        for line in strain_lines:
            num_str = re.findall(r'[-+\d\.Ee]+', line)  # the expression matches one or more repetitions of "-", "integer", ".", "E",
            num = float(num_str[1])
            strains_one_step.append(num)
        strains.append(strains_one_step)     
    
    return np.array(trajectory).astype(float), np.array(particle_types).astype(float), np.array(strains).astype(float)


def parse_simulation_strain(file):
    '''
    Extract max-principal strain (mps).
    '''
    # append _strain to the file (C_80_320_Aa_40.txt --> C_80_320_Aa_40_strain.txt)
    filename = file.split('.')[0] + '_strain.txt'
    filename = filename.replace('coordinates_eps', 'mps') # subdir from coordinates_eps to mps
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find all "particle position" lines and "plastic strain" lines using key words
    strain_lines_start, strain_lines_end = [], []
    for idx, line in enumerate(lines):
        if line.startswith("$RESULT OF  Max Prin Strain"):
            strain_lines_start.append(idx)
        elif line.startswith("*END"):  
            strain_lines_end.append(idx)
            
    # Extrac effective plastic strain
    strains = []
    for line_start, line_end in zip(strain_lines_start, strain_lines_end):
        strain_lines = lines[line_start+1:line_end]   # lines that contains positions in one time step
        strains_one_step = []
        for line in strain_lines:
            num_str = re.findall(r'[-+\d\.Ee]+', line)  # the expression matches one or more repetitions of "-", "integer", ".", "E",
            num = float(num_str[1])
            strains_one_step.append(num)
        strains.append(strains_one_step) 
        
    return np.array(strains).astype(float)


if __name__ == "__main__":
    pass