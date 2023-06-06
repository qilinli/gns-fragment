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
MAX_I = np.array([325, 150])
MIN_I = np.array([-325, -30])
MAX_C = np.array([325, 95])
MIN_C = np.array([-325, -15])
strain_min = 0
strain_max = 2

TYPE_TO_COLOR = {
    3: "red",  # Kinematic (actuator in 2d-C)
    2: "lightsteelblue",  # steel particle (actuator in 2D-I)
    1: "green", # support particle
    # 0: "blue"  # concrete particle
}

def main(unused_argv):   
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(FLAGS.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)
    
    if 'Concrete2D-C' in FLAGS.rollout_path:
        MAX, MIN = MAX_C, MIN_C
    else:
        MAX, MIN = MAX_I, MIN_I
    y_scaling_factor = (MAX - MIN)[0] / (MAX - MIN)[1]

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), gridspec_kw={"height_ratios":[20,20,2]})
    
    plot_info = []
    for ax_i, (label, rollout_field) in enumerate(
        [("LS-DYNA", "ground_truth_rollout"),
        ("GNN", "predicted_rollout")]):
    
        # Append the initial positions to get the full trajectory.
        trajectory = np.concatenate([
            rollout_data["initial_positions"],
            rollout_data[rollout_field]], axis=0)
        trajectory[:,:,1] = trajectory[:,:,1] * y_scaling_factor # if xy-scaled
        trajectory = trajectory * (MAX - MIN) + MIN ## qilin 
        if label == 'LS-DYNA':
            trajectory_gt = trajectory
        
        if label == "LS-DYNA":
            strain = rollout_data["ground_truth_strain"]
            strain_gt = strain
        elif label == "GNN":
            strain = rollout_data["predicted_strain"]
        strain = np.concatenate((rollout_data["initial_strains"], strain), axis=0)
        #strain = strain * (strain_max - strain_min) + strain_min   ## inverse normalisation
        # strain = strain * rollout_data["metadata"]["strain_std"] + rollout_data["metadata"]["strain_mean"]  # if strain standardised
        print(strain.min(), strain.max())
        
        x_min, y_min = trajectory_gt.min(axis=(0,1))
        x_max, y_max = trajectory_gt.max(axis=(0,1))

        ax = axes[ax_i]
        ax.set_title(label)
        bounds = rollout_data["metadata"]["bounds"]

        ax.set_xlim(x_min-10, x_max+10)  ## qilin
        ax.set_ylim(y_min-10, y_max+10)   ## qilin

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
        
        cmap = matplotlib.cm.rainbow
        norm = matplotlib.colors.Normalize(vmin=strain_gt.min(axis=(0,1)), vmax=strain_gt.max(axis=(0,1)))
        cb = matplotlib.colorbar.ColorbarBase(axes[-1], cmap=cmap, norm=norm, orientation='horizontal')
        cb_label = 'Maximal principal strain' if 'mps' in FLAGS.rollout_path else 'Effective plastic strain'
        cb.set_label(cb_label)
                     
        concrete_points = ax.scatter([], [], c=[], s=6, cmap="rainbow", vmin=strain_gt.min(axis=(0,1)), vmax=strain_gt.max(axis=(0,1)))
        other_points = {particle_type: ax.plot([], [], "o", ms=3, color=color)[0] for particle_type, color in TYPE_TO_COLOR.items()}

        plot_info.append((trajectory, strain, concrete_points, other_points))

    num_steps = trajectory.shape[0]   
    
    def update(step_i):
        frames_to_save = []   # corresponds to 0ms, 100ms, 200ms, 300ms, 400ms, 500ms
        outputs = []
        for trajectory, strain, concrete_points, other_points in plot_info:
            concrete_points.set_offsets(trajectory[step_i, :])
            concrete_points.set_array(strain[step_i,:])
            outputs.append(concrete_points)
            for particle_type, line in other_points.items():
                mask = rollout_data["particle_types"] == particle_type
                line.set_data(trajectory[step_i, mask, 0], trajectory[step_i, mask, 1])
                outputs.append(line)
        if step_i in frames_to_save: plt.savefig(FLAGS.output_path.replace('.gif', f'_frame{step_i}.png'), dpi=80)
        return outputs

    unused_animation = animation.FuncAnimation(
        fig, update,
        frames=np.arange(0, num_steps, FLAGS.step_stride), interval=50) # 'interval': Delay between frames in milliseconds.

    unused_animation.save(FLAGS.output_path, dpi=100, fps=10, writer='pillow')   # animation length = num_steps / fps 
    # unused_animation.save(FLAGS.output_path, dpi=300, fps=60, writer='FFMpeg')
    # plt.show(block=FLAGS.block_on_show)
  

if __name__ == "__main__":
    app.run(main)
