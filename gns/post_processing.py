import pickle
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import trimesh
from matplotlib import cm, colors
import datetime
from pathlib import Path
import pandas as pd
import seaborn as sns


VELOCITY_SCALE_FACTOR = 100 / 6  # for stepsize = 0.06 ms
DIST_THRES = 10.12 # the smaller the more fragments
MASS_PER_PARTICLE = 0.0024
MAX_FRAGMENT_SIZE = 100
n_steps = 100

def compute_particle_mask(init_particle_pos, charge_weight):
    
    thres = np.sqrt(charge_weight) * 150
    center_mask =  (init_particle_pos[:, 0] < thres) & (init_particle_pos[:, 0] > -thres) & (init_particle_pos[:, 1] < thres) & (init_particle_pos[:, 1] > -thres)
    
    return center_mask
    
def compute_fragment(particle_position, dist_thres=10.2, max_fragment_size=100):
    kdt = KDTree(particle_position)
    indices = kdt.query_radius(particle_position, r=dist_thres)
    visited = set()
    fragments = []
    particles_in_fragments = set()

    for idx, neighbors in enumerate(indices):
        if idx not in visited and idx not in particles_in_fragments:
            new_fragment = set()
            stack = [idx]
            while stack:
                current = stack.pop()
                if current not in visited and current not in particles_in_fragments:
                    visited.add(current)
                    new_fragment.add(current)
                    stack.extend([n for n in indices[current] if n not in visited and n not in particles_in_fragments])

            if len(new_fragment) <= MAX_FRAGMENT_SIZE:
                fragments.append(new_fragment)
                particles_in_fragments.update(new_fragment)
                
    
    return  fragments

def compute_fragment_property(particle_position, particle_last_position, fragments):
    centres, masses, diameters, vels = [], [], [], []
    
    for idx, fragment in enumerate(fragments):
        fragment_positions = particle_position[list(fragment)] 
        fragment_centre = fragment_positions.mean(axis=0)
        fragment_mass = len(fragment)*MASS_PER_PARTICLE

        # calculate spatial size (diameter of the fragment)
        if len(fragment) >= 2:
            distances = pairwise_distances(fragment_positions, fragment_positions)
            fragment_diameter = distances.max()
        else:
            fragment_diameter = 10  # single element diameter

        # calculate fragment speed
        particle_vel = particle_position - particle_last_position
        fragment_vels = particle_vel[list(fragment)] * VELOCITY_SCALE_FACTOR
        fragment_vel = np.mean(fragment_vels, axis=0)
        
        centres.append(fragment_centre)
        masses.append(fragment_mass)
        diameters.append(fragment_diameter)
        vels.append(fragment_vel)
    
    # Conver list to np array
    centres = np.array(centres)
    masses = np.array(masses)
    diameters = np.array(diameters)
    vels = np.array(vels)
    
    return centres, masses, diameters, vels


def save_property_csv(fragments_centre, fragments_mass, fragments_diameter, fragments_vel, case, step, savename):
    metadata = {
        'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Case': case,
        'Step': step,      
    }
    metadata_str = "\n".join([f"# {key}: {value}" for key, value in metadata.items()]) + '\n'
    
    data = {
        'Centre X': fragments_centre[:, 0],
        'Centre Y': fragments_centre[:, 1],
        'Centre Z': fragments_centre[:, 2],
        'Mass': fragments_mass,
        'Diameter': fragments_diameter,
        'Velocity X': fragments_vel[:, 0],
        'Velocity Y': fragments_vel[:, 1],
        'Velocity Z': fragments_vel[:, 2],
    }
    df = pd.DataFrame(data)
    
    # Save DataFrame to CSV with metadata
    with open(savename, 'w') as f:
        f.write(metadata_str)
    df.to_csv(savename, index=False, mode='a', header=True)
    
def compute_mass_distribution(fragments_mass, fragments_diameter, cut_thres=[10, 30, 50, 80]):
    mass_distribution = []
    
    # the first range
    mask = fragments_diameter < cut_thres[0]
    mass = fragments_mass[mask].sum()
    mass_distribution.append(mass)
    
    # subsequent range
    for i in range(len(cut_thres)-1):
        mask = (fragments_diameter >= cut_thres[i]) & (fragments_diameter < cut_thres[i+1])
        mass = fragments_mass[mask].sum()
        mass_distribution.append(mass)
    
    # the last range
    mask = fragments_diameter >= cut_thres[-1]
    mass = fragments_mass[mask].sum()
    mass_distribution.append(mass)
    
    return mass_distribution

def plot_mass_distribution_bar(mass_distribution, savename):
    x_labels = ['0-10', '10-30', '30-50', '50-80', '>80']
    
    # Create a DataFrame to hold the data
    df = pd.DataFrame({
        'Categories': x_labels,
        'Values': mass_distribution,
        'Data': ['dummy']*len(mass_distribution)
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    # Use Seaborn's barplot
    sns.set_theme(style='ticks')
    sns.barplot(data=df, x='Categories', y='Values', hue='Data')

    # Adjusting font size of tick labels and axis labels
    ax.tick_params(labelsize=20)
    ax.set_ylabel('Mass (kg)', fontsize=24)
    ax.set_xlabel('Fragment size (mm)', fontsize=24)
    ax.legend_.remove()
    ax.grid(True, linestyle='--')

    # Adaptive ylim based on mass_distribution values
    max_value = max(mass_distribution)
    ax.set_ylim([0, max_value + (0.1 * max_value)])  # Add 10% to the max value for spacing

    # Add title with mass_distribution value with .3f precision
    formatted_values = ', '.join([f'{val:.2f}' for val in mass_distribution])
    title_str = f"Mass Distribution Bar Plot - Values: {formatted_values}"
    ax.set_title(title_str, fontsize=20)

    # Save the plot
    plt.savefig(savename, bbox_inches='tight')
    plt.close()

def plot_eps(particle_pos, particle_strain, particle_type, eps_bug_mask, case, view, savename):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Apply mask for particle type
    mask = particle_type == 1
    particle_strain[mask] = 0

    # Scatter plot colored by strain values
    sc = ax.scatter(particle_pos[~eps_bug_mask, 0], particle_pos[~eps_bug_mask, 1], particle_pos[~eps_bug_mask, 2], s=2, c=particle_strain[~eps_bug_mask], vmin=0, vmax=2, cmap='jet')

    # Setting the aspect ratio of the plot to be equal
    ax.set_box_aspect([np.ptp(a) for a in particle_pos.T])

    # Remove grid, ticks, and labels
    ax.set_title(case)
    ax.grid(False)
    ax.set_zticklabels([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('')

    # Hide the Z-axis
    ax.zaxis.line.set_lw(0)
    ax.set_zticks([])

    # Adjust the margin between the axis ticks and labels
    ax.tick_params(axis='both', which='major', pad=8)

    # Adding a colorbar
    cbar = plt.colorbar(sc, orientation='horizontal', pad=0., aspect=70)
    cbar.set_label('Effective Plastic Strain')

    # Changing the view angle
    elev = -90 if 'bot' in view else 90
    ax.view_init(elev=elev, azim=0)

    # Save the plot to a file
    plt.savefig(savename)
    plt.close()

def plot_fragment(masked_particle_position, fragments, fragments_vel, case, savename):
    positions = masked_particle_position
    fragments_vel = np.linalg.norm(fragments_vel, axis=1)
    norm = colors.Normalize(vmin=np.min(fragments_vel), vmax=np.max(fragments_vel))
    cmap = cm.jet
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=0.5, c='grey', alpha=0.3)

    # Loop over fragments and plot each mesh
    for idx, fragment in enumerate(fragments):
        fragment_positions = positions[list(fragment)]
        fragment_vel = fragments_vel[idx]

        if len(fragment) > 3:
            mesh = trimesh.Trimesh(vertices=fragment_positions, process=True)
            hull = mesh.convex_hull

            # Get the vertices and faces from the mesh
            vertices = hull.vertices
            faces = hull.faces
            color = cmap(norm(fragment_vel))

            # Create a Poly3DCollection from the vertices and faces
            mesh_plot = Poly3DCollection(vertices[faces], edgecolor='k', facecolors=color, linewidths=0.1, alpha=0.9)
            ax.add_collection3d(mesh_plot)

    # Set plot limits
    ax.set_box_aspect([np.ptp(a) for a in positions.T])
    ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
    ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())
    ax.set_zlim(positions[:, 2].min(), positions[:, 2].max())

    # Set labels
    ax.set_title(case)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add color bar
    ax_colorbar = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(fragments_vel)
    cbar = plt.colorbar(mappable, shrink=0.8, aspect=50, cax=ax_colorbar)
    cbar.set_label('fragment velocity (m/s)')
    
    # Save the plot to a file
    plt.savefig(savename)
    plt.close()

def compute_zero_eps_mask(particle_trajectories, particle_strains, particle_type):
    xy_mask = (particle_trajectories[0, :, 0] < 250)&(particle_trajectories[0, :, 0] > -250)&(particle_trajectories[0, :, 1] < 250)&(particle_trajectories[0, :, 1] > -250)
    type_mask = (particle_type == 0) 
    strain_mask = (particle_strains[-1, :] < 1)
    eps_bug_mask = xy_mask & type_mask & strain_mask
    return eps_bug_mask

def compute_max_vel(fragments_vel):
    return (np.linalg.norm(fragments_vel, axis=1)).max()

def main(sample_path, charge_weight, particle_trajectories, particle_strains, particle_type, rollout_step=81):
    root_dir = Path(sample_path).parent.parent
    case_name = Path(sample_path).parent.name
    output_path = root_dir / 'output' / case_name
    property_dir = output_path / 'property'
    Path(property_dir).mkdir(parents=True, exist_ok=True)

    mass_dir = output_path / 'mass'
    Path(mass_dir).mkdir(parents=True, exist_ok=True)

    eps_dir = output_path / 'eps'
    Path(eps_dir).mkdir(parents=True, exist_ok=True)

    fragment_dir = output_path / 'fragment'
    Path(fragment_dir).mkdir(parents=True, exist_ok=True)
    
    mask = compute_particle_mask(particle_trajectories[0], charge_weight)
    eps_bug_mask = compute_zero_eps_mask(particle_trajectories, particle_strains, particle_type)
    
    for step in range(10, rollout_step, 10):
        masked_particle_position = particle_trajectories[step, mask]
        masked_particle_previous_position = particle_trajectories[step-1, mask]
        fragments = compute_fragment(masked_particle_position, dist_thres=10.2, max_fragment_size=100)
        fragments_centre, fragments_mass, fragments_diameter, fragments_vel = compute_fragment_property(masked_particle_position, masked_particle_previous_position, fragments)
        try:
            save_property_csv(fragments_centre, fragments_mass, fragments_diameter, fragments_vel, 
                              case=output_path.name, step=step, savename=str(property_dir / f'fragments_properties_step_{step}.csv'))
            mass_distribution = compute_mass_distribution(fragments_mass, fragments_diameter)                                    
            plot_mass_distribution_bar(mass_distribution, savename=str(mass_dir/ f'mass_step_{step}'))
            plot_eps(particle_trajectories[step], particle_strains[step], particle_type, eps_bug_mask, case=case_name, view='bot', savename=str(eps_dir/ f'eps_bot_step_{step}'))
            plot_eps(particle_trajectories[step], particle_strains[step], particle_type, eps_bug_mask, case=case_name, view='top', savename=str(eps_dir/ f'eps_top_step_{step}'))
            plot_fragment(masked_particle_position, fragments, fragments_vel, case=case_name, savename=str(fragment_dir/ f'fragment_step_{step}'))
        
        except IndexError:
            print(f"No fragments, skipping {case_name}")
            continue
        

if __name__ == '__main__':
    # Define the conditions for fragment filtering
    start_time = time.time()
    case = 'd3plot4'
    path = f'/home/jovyan/work/gns-fragment/rollouts/Fragment/inference/input0.4-1ms/{case}.pkl'

    case_dir = Path(f'/home/jovyan/work/gns-fragment/rollouts/Fragment/inference/temp/{case}_0.4-1ms/')
    property_dir = case_dir / 'property'
    Path(property_dir).mkdir(parents=True, exist_ok=True)

    mass_dir = case_dir / 'mass'
    Path(mass_dir).mkdir(parents=True, exist_ok=True)

    eps_dir = case_dir / 'eps'
    Path(eps_dir).mkdir(parents=True, exist_ok=True)

    fragment_dir = case_dir / 'fragment'
    Path(fragment_dir).mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "rb") as file:
            rollout_data = pickle.load(file)
    except FileNotFoundError:
        print(f"File {path} not found.")

    pred_trajs = rollout_data['pred_trajs']
    particle_strains = rollout_data['pred_strains']
    particle_type = rollout_data['particle_type']
   
    main(case, particle_trajectories, particle_strains, particle_type, rollout_step=81)

    elapsed_time = time.time() - start_time
    print(f"Took {elapsed_time} to finish {case}")