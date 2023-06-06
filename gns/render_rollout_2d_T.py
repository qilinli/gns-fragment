# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simple matplotlib rendering of a rollout prediction against ground truth.

Usage (from parent directory):

`python -m gns.render_rollout --rollout_path={OUTPUT_PATH}/rollout_test_1.pkl --output_path={OUTPUT_PATH}/rollout_test_1.gif`

Where {OUTPUT_PATH} is the output path passed to `train.py` in "eval_rollout"
mode.

It may require installing Tkinter with `sudo apt-get install python3.7-tk`.

"""  # pylint: disable=line-too-long

import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")
flags.DEFINE_string("output_path", None, help="Path to output fig file")
flags.DEFINE_integer("step_stride", 1, help="Stride of steps to skip.")

FLAGS = flags.FLAGS

# Qilin, concrete stats for inverse nomalisation
MAX, MIN = np.array([100, 50]), np.array([-2.5, -50])
strain_mean, strain_std = 143.09920564186177, 86.05175002337249  # vms stats pre-computed from data

TYPE_TO_COLOR = {
    3: "red",  # Kinematic (actuator in 2d-C)
    2: "lightsteelblue",  # steel particle (actuator in 2D-I)
    #1: "blue", # support particle
    # 0: "blue"  # concrete particle
}

def main(unused_argv):   
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(FLAGS.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10), gridspec_kw={"width_ratios":[10,10,0.5]})
    
    plot_info = []
    for ax_i, (label, rollout_field) in enumerate(
        [("LS-DYNA", "ground_truth_rollout"),
        ("GNN", "predicted_rollout")]):
    
        # Append the initial positions to get the full trajectory.
        trajectory = np.concatenate([
            rollout_data["initial_positions"],
            rollout_data[rollout_field]], axis=0)

        trajectory = trajectory * (MAX - MIN) + MIN ## qilin 
        if label == 'LS-DYNA':
            trajectory_gt = trajectory
        
        if label == "LS-DYNA":
            strain = rollout_data["ground_truth_strain"]
            strain_gt = strain * strain_std + strain_mean  ## inverse normalisation
        elif label == "GNN":
            strain = rollout_data["predicted_strain"]
        strain = np.concatenate((rollout_data["initial_strains"], strain), axis=0)
        strain = strain * strain_std + strain_mean  ## inverse normalisation
         
        x_min, y_min = trajectory_gt.min(axis=(0,1))
        x_max, y_max = trajectory_gt.max(axis=(0,1))
        
        ax = axes[ax_i]
        ax.set_title(label)
        bounds = rollout_data["metadata"]["bounds"]

        ax.set_xlim(x_min-5, x_max+5)  ## qilin
        ax.set_ylim(y_min-5, y_max+5)   ## qilin
    
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
        
        cmap = matplotlib.cm.rainbow
        norm = matplotlib.colors.Normalize(vmin=strain_gt.min(axis=(0,1)), vmax=strain_gt.max(axis=(0,1)))
        cb = matplotlib.colorbar.ColorbarBase(axes[-1], cmap=cmap, norm=norm, orientation='vertical')
        cb_label = 'Von mises stress (MPa)'
        cb.set_label(cb_label, fontsize=15)
        cb.ax.tick_params(labelsize=12)
        
        concrete_points = ax.scatter([], [], c=[], s=6, cmap="rainbow", vmin=strain_gt.min(axis=(0,1)), vmax=strain_gt.max(axis=(0,1)))
        other_points = {particle_type: ax.plot([], [], "o", ms=3, color=color)[0] for particle_type, color in TYPE_TO_COLOR.items()}

        plot_info.append((trajectory, strain, concrete_points, other_points))

    num_steps = trajectory.shape[0]   
    
    def update(step_i):
        frames_to_save = [30, 60, 90]   # corresponds to 0ms, 100ms, 200ms, 300ms, 400ms, 500ms
        outputs = []
        for trajectory, strain, concrete_points, other_points in plot_info:
            concrete_points.set_offsets(trajectory[step_i, :])
            concrete_points.set_array(strain[step_i,:])
            outputs.append(concrete_points)
            for particle_type, line in other_points.items():
                mask = rollout_data["particle_types"] == particle_type
                line.set_data(trajectory[step_i, mask, 0], trajectory[step_i, mask, 1])
                outputs.append(line)
        if step_i in frames_to_save: plt.savefig(FLAGS.output_path.replace('.gif', f'_frame{step_i}.png'), dpi=100)
        return outputs

    unused_animation = animation.FuncAnimation(
        fig, update,
        frames=np.arange(0, num_steps, FLAGS.step_stride), interval=50) # 'interval': Delay between frames in milliseconds.

    unused_animation.save(FLAGS.output_path, dpi=100, fps=5, writer='pillow')   # animation length = num_steps / fps 
    # unused_animation.save(FLAGS.output_path, dpi=300, fps=60, writer='FFMpeg')
    # plt.show(block=FLAGS.block_on_show)
  

if __name__ == "__main__":
    app.run(main)
