import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os


flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout `.pkl` file")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.")
flags.DEFINE_enum("output_mode", "gif", ["gif", "vtk"], help="Type of render output")
flags.DEFINE_enum("dataset", "RC", ["RC", "Fragment"], help="Which data to render")

FLAGS = flags.FLAGS
    
class Render():

    def __init__(self, input_dir, input_name):
        # Texts to describe rollout cases for data and render          
        rollout_cases = [
            ["ground_truth_rollout", "LS-DYNA"], ["predicted_rollout", "GNN"]]
        strain_cases = [
            ["ground_truth_strain", "LS-DYNA"], ["predicted_strain", "GNN"]]
        self.rollout_cases = rollout_cases
        self.strain_cases = strain_cases
        self.input_dir = input_dir
        self.input_name = input_name
        self.output_dir = input_dir
        self.output_name = input_name

        # Get trajectory
        with open(f"{self.input_dir}{self.input_name}.pkl", "rb") as file:
            rollout_data = pickle.load(file)
        self.rollout_data = rollout_data
        trajectory = {}
        for rollout_case in rollout_cases:
            trajectory[rollout_case[0]] = np.concatenate(
                [rollout_data["initial_positions"], rollout_data[rollout_case[0]]], axis=0
            )
        strain = {}
        for strain_case in strain_cases:
            strain[strain_case[0]] = np.concatenate(
                [rollout_data["initial_strains"], rollout_data[strain_case[0]]], axis=0
            )
        # for k, v in strain.items():
        #     strain[k] = v * STRAIN_STD + STRAIN_MEAN
        # print(strain["predicted_strain"].mean(), strain["predicted_strain"].std(), 
        #       strain["predicted_strain"].min(), strain["predicted_strain"].max())
        # print(strain["ground_truth_strain"].mean(), strain["ground_truth_strain"].std(),
        #       strain["ground_truth_strain"].min(), strain["ground_truth_strain"].max())
        
        # Trajectory information
        self.trajectory = trajectory
        self.strain = strain
        self.dims = trajectory[rollout_cases[0][0]].shape[2]
        self.num_particles = trajectory[rollout_cases[0][0]].shape[1]
        self.num_steps = trajectory[rollout_cases[0][0]].shape[0]
        self.boundaries = rollout_data["metadata"]["bounds"]
        self.particle_type = rollout_data["particle_types"]

    def color_map(self, datacase):
        """
        Create a colormap for each timestep based on strain
        """
        color_map = []
        for t in range(self.num_steps):
            normalized_strain = mcolors.Normalize(
                vmin=self.strain["ground_truth_strain"].min(), 
                vmax=self.strain["ground_truth_strain"].max()
            )
            colormap = plt.get_cmap('jet')
            color_map.append(colormap(normalized_strain(self.strain[datacase][t])))
        return color_map

    def render_gif_animation(
            self, point_size=1, timestep_stride=3, vertical_camera_angle=20, viewpoint_rotation=0.5, roll=None
    ):
        """
        Render `.gif` animation from `,pkl` trajectory.
        :param point_size: particle size for visualization
        :param timestep_stride: numer of timesteps to stride for visualization (i.e., sampling rate)
        :param vertical_camera_angle: camera angle in airplane view in 3d render
        :param viewpoint_rotation: speed of viewpoint rotation in 3d render
        :return: gif format animation
        """
        # Init figures
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(2, 1, 1, projection='3d')
        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        axes = [ax1, ax2]

        # Define datacase name
        trajectory_datacases = [self.rollout_cases[0][0], self.rollout_cases[1][0]]
        render_datacases = [self.rollout_cases[0][1], self.rollout_cases[1][1]]

        # Get boundary of simulation
        xboundary = self.boundaries[0]
        yboundary = self.boundaries[1]
        zboundary = self.boundaries[2]
        for ax in axes:
            ax.set_box_aspect([xboundary[1] - xboundary[0], 
                               yboundary[1] - yboundary[0], 
                               zboundary[1] - zboundary[0]])
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # Fig creating function for 3d
        def animate(i):
            print(f"Render step {i}/{self.num_steps} for {self.output_name}")

            fig.clear()
            for j, datacase in enumerate(trajectory_datacases):
                axes[j] = fig.add_subplot(2, 1, j+1, projection='3d')
                axes[j].set_box_aspect([xboundary[1] - xboundary[0], 
                                       yboundary[1] - yboundary[0], 
                                       zboundary[1] - zboundary[0]])
                if render_datacases[j] == "LS-DYNA":
                    color_map = self.color_map('ground_truth_strain')
                else:
                    color_map = self.color_map('predicted_strain')
                
                axes[j].set_xlabel('x')
                axes[j].set_ylabel('y')
                axes[j].set_zlabel('z')
                axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
                axes[j].set_zlim([float(zboundary[0]), float(zboundary[1])])
                axes[j].scatter(self.trajectory[datacase][i, :, 0],
                                self.trajectory[datacase][i, :, 1],
                                self.trajectory[datacase][i, :, 2], 
                                s=point_size, 
                                color=color_map[i]
                               )
                # rotate viewpoints angle little by little for each timestep
                axes[j].view_init(elev=vertical_camera_angle, azim=i*viewpoint_rotation, roll=roll, vertical_axis='z')
                axes[j].grid(True, which='both')
                axes[j].set_title(render_datacases[j])
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # Creat animation
        ani = animation.FuncAnimation(
            fig, animate, frames=np.arange(0, self.num_steps, timestep_stride), interval=10)

        ani.save(f'{self.output_dir}{self.output_name}.gif', dpi=100, fps=10, writer='Pillow')
        print(f"Animation saved to: {self.output_dir}{self.output_name}.gif")


def main(_):
    if not FLAGS.rollout_dir:
        raise ValueError("A `rollout_dir` must be passed.")
    if not FLAGS.rollout_name:
        raise ValueError("A `rollout_name`must be passed.")
        
    if FLAGS.dataset == 'RC':
        vertical_camera_angle = 20
        viewpoint_rotation = 0
        roll = 0
        STRAIN_MEAN, STRAIN_STD = 0.5426821307442955, 0.7741500060379738
    else:
        vertical_camera_angle = 10
        viewpoint_rotation = 0.5
        roll = 0
        STRAIN_MEAN, STRAIN_STD = 0.8868453123315391, 0.6590170029193022
            
    render = Render(input_dir=FLAGS.rollout_dir, input_name=FLAGS.rollout_name)
    
    render.render_gif_animation(
        point_size=1,
        timestep_stride=FLAGS.step_stride,
        vertical_camera_angle=vertical_camera_angle,
        viewpoint_rotation=viewpoint_rotation,
        roll=roll
    )

if __name__ == '__main__':
    app.run(main)

