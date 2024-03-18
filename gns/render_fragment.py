import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from matplotlib import cm, colors
from absl import app
from absl import flags
from PIL import Image
import glob
import plotly.graph_objects as go
import plotly.io as pio

flags.DEFINE_string("rollout_dir", '/home/jovyan/work/gns-fragment/rollouts/Fragment/Step-0-100-3-AllTest', help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", 'rollout_0', help="Name of rollout `.pkl` file")
flags.DEFINE_integer("step_stride", 1, help="Stride of steps to skip.")

FLAGS = flags.FLAGS

MASS_PER_PARTICLE = 0.0024
VELOCITY_SCALE_FACTOR = 100 / 6
MAX_FRAGMENT_SIZE = 100
DIST_THRES = 10.12 


def load_data_from_pkl(rollout_dir, rollout_name):
    charge_weight = int(rollout_name.split('_')[2])

    with open(Path(rollout_dir) / (rollout_name + '.pkl'), "rb") as file:
        rollout_data = pickle.load(file)

    init_pos = rollout_data['initial_positions']
    pred_pos = rollout_data['predicted_rollout']
    gt_pos = rollout_data['ground_truth_rollout']
    gt_pos = np.concatenate((init_pos, gt_pos), axis=0)
    pred_pos = np.concatenate((init_pos, pred_pos), axis=0)
    
    return gt_pos, pred_pos, charge_weight


def compute_particle_mask(init_pos, charge_weight):
    thres = np.sqrt(charge_weight) * 150
    center_mask =  (init_pos[:, 0] < thres) & (init_pos[:, 0] > -thres) & (init_pos[:, 1] < thres) & (init_pos[:, 1] > -thres)
    
    return center_mask


def compute_fragment_by_tree(particle_pos, dist_thres=10.12, max_fragment_size=100):
    kdt = KDTree(particle_pos)
    indices = kdt.query_radius(particle_pos, r=dist_thres)
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

            if len(new_fragment) <= max_fragment_size:
                fragments.append(new_fragment)
                particles_in_fragments.update(new_fragment)
                
    return fragments


def plot_fragment_by_plotly(particle_pos, particle_vel, fragments, xyz_min, xyz_max, step):
    xmin, ymin, zmin = -300, -300, 0
    xmax, ymax, zmax = 300, 300, 300

    fig = go.Figure(data=[go.Scatter3d(
            x=particle_pos[:, 0],
            y=particle_pos[:, 1],
            z=particle_pos[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=particle_vel,  # Use the normalized velocities here
                colorscale='Jet',
                cmin=0,
                cmax=100,
                opacity=0.3
            )
        )])

    # Counter for number of fragments
    num_fragment_particles = 0

    # Loop over fragments
    for i, fragment in enumerate(fragments):
        fragment_positions = particle_pos[list(fragment)]
        fragment_velocities = particle_vel[list(fragment)]
        num_fragment_particles += len(fragment)

        # Check if this fragment satisfies the conditions
        if len(fragment) > 3:
            # Compute the average normalized velocity for this fragment
            avg_velocity = np.mean(fragment_velocities)

            # Map the average velocity to a color
            color_rgb = cm.jet(avg_velocity / 100)[:3] # changed to jet colormap
            color_rgba = f"rgba({color_rgb[0]*255}, {color_rgb[1]*255}, {color_rgb[2]*255}, 0.8)"

            mesh = trimesh.Trimesh(vertices=fragment_positions, process=True)
            hull = mesh.convex_hull

            # Then add the mesh to the figure
            fig.add_trace(go.Mesh3d(
                x=hull.vertices[:, 0],
                y=hull.vertices[:, 1],
                z=hull.vertices[:, 2],
                i=hull.faces[:, 0],
                j=hull.faces[:, 1],
                k=hull.faces[:, 2],
                color=color_rgba,
                intensity=[avg_velocity]*hull.faces.shape[0],
                colorscale='Jet',
                cmin=0,
                cmax=100,
                showscale=True,
            ))
    fragment_mass = num_fragment_particles * 0.0024

    fig.update_scenes(
        xaxis=dict(range=[xmin, xmax]), 
        yaxis=dict(range=[ymin, ymax]), 
        zaxis=dict(range=[zmin, zmax])
    )

    # Update layout with the title
    fig.update_layout(
        autosize=False,
        width=1920,
        height=1080,
        scene=dict(
            xaxis=dict(title='X', title_font=dict(size=22), tickfont=dict(size=16)),
            yaxis=dict(title='Y', title_font=dict(size=22), tickfont=dict(size=16)),
            zaxis=dict(title='Z', title_font=dict(size=22), tickfont=dict(size=16)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            camera = dict(
                up=dict(x=0, y=0, z=1),  # this is the 'up' direction for the camera
                center=dict(x=0, y=0, z=0),  # this will move the camera itself
                eye=dict(x=1.5, y=1.5, z=0.3)  # this moves the 'eye' of the camera
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # tight layout
        title=dict(
            text=f"Step: {step:02}, time: {step*0.06:.2f} ms, fragment mass: {fragment_mass:.3f} kg", 
            x=0.48,
            y=0.75,
            xanchor='center',
            yanchor='top',
            font=dict(
                size=35,  # Adjust the font size here
                family="Courier New, monospace",  # Optional: specify font family
            )
        ),
    )
    
    return fig

def imgs_to_gif(out_path, mode):
    # List of images
    img_paths = glob.glob(str(out_path / f'{mode}*.png'))
    img_paths.sort()

    # Read in images
    imgs = [Image.open(img_path) for img_path in img_paths]

    # Create a new image object for the first frame, then append the remaining frames.
    imgs[0].save(str(out_path / f'{mode}.gif'), format='GIF', append_images=imgs[1:], save_all=True, duration=500, loop=0)
    
    print(f"Saved {str(out_path / f'{mode}.gif')}")
    
def main(_):
    gt_pos, pred_pos, charge_weight = load_data_from_pkl(FLAGS.rollout_dir, FLAGS.rollout_name)
    init_pos = gt_pos[0, :]
    last_pos = gt_pos[-1, :]
    xyz_min = last_pos.min(axis=0)
    xyz_max = last_pos.max(axis=0)
    mask = compute_particle_mask(init_pos, charge_weight)
    
    for mode in ['gt', 'pred']:
        if mode == 'gt':
            pos = gt_pos
        else:
            pos = pred_pos
            
        for step in range(1, 34, FLAGS.step_stride):
            current_pos = pos[step, :]
            previous_pos = pos[step-1, :]

            current_pos_masked = current_pos[mask]
            previous_pos_masked = previous_pos[mask]
            particles_vel = np.linalg.norm(current_pos_masked - previous_pos_masked, axis=1) * VELOCITY_SCALE_FACTOR 

            fragments = compute_fragment_by_tree(current_pos_masked)

            fig = plot_fragment_by_plotly(current_pos_masked, particles_vel, fragments, xyz_min, xyz_max, step)

            file_name = f"{mode}-{step:02}.png"
            out_path = Path(FLAGS.rollout_dir) / 'fragment' / FLAGS.rollout_name
            out_path.mkdir(parents=True, exist_ok=True)
            save_path = out_path / file_name

            fig.write_image(str(save_path), scale=2)
            
        imgs_to_gif(out_path, mode)

        
if __name__ == '__main__':
    app.run(main)