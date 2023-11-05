import collections
import json
import numpy as np
import os
import os.path as osp
import sys
import torch
import pickle
import glob
import re
import tree
import time
from absl import flags
from absl import app

from gns import learned_simulator
from gns import reading_utils
from gns import post_processing



flags.DEFINE_string('data_path', r'C:\Users\kylin\OneDrive - Curtin\research\civil_engineering\data\FGN\C30_120_6_0.4', 
                    help='The dataset directory.')
flags.DEFINE_string('model_path', './models/Fragment/Benchmark-NS5e-4_1e-2_R14_L5N64_PosNsx10/', help=(
    'The path for saving checkpoints of the model.'))
flags.DEFINE_string('model_file', 'model-066000.pt', help=(
    'Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('output_path', 'rollouts/Fragment/inference/', help='The path for saving outputs (e.g. rollouts).')

# Model parameters
flags.DEFINE_float('connection_radius', 14, help='connectivity radius for graph.')
flags.DEFINE_integer('layers', 5, help='Number of GNN layers.')
flags.DEFINE_integer('hidden_dim', 64, help='Number of neurons in hidden layers.')
flags.DEFINE_integer('dim', 3, help='The dimension of concrete simulation.')
flags.DEFINE_float('noise_std', 5e-4, help='The std deviation of the noise.')


FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 10  # So we can calculate the last 9 velocities.
NUM_PARTICLE_TYPES = 3     # concrete 0, rebar 1
REBAR_PARTICLE_ID = 1
BOUNDARY_PARTICLE_ID = 2

def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: str) -> learned_simulator.LearnedSimulator:
    """Instantiates the simulator.
  
    Args:
      metadata: JSON object with metadata.
      acc_noise_std: Acceleration noise std deviation.
      vel_noise_std: Velocity noise std deviation.
      device: PyTorch device 'cpu' or 'cuda'.
    """
    acc_noise_std = torch.FloatTensor([0.0005, 0.0005, 0.01])
    vel_noise_std = torch.FloatTensor([0.0005, 0.0005, 0.01])
    pos_noise_std = torch.FloatTensor([0.005, 0.005, 0.05])
    
    
    acc_mean = torch.FloatTensor(metadata['acc_mean']).to(device)
    acc_std = torch.sqrt(torch.FloatTensor(metadata['acc_std']) ** 2 + acc_noise_std ** 2).to(device)

    
    vel_mean = torch.FloatTensor(metadata['vel_mean']).to(device)
    vel_std = torch.sqrt(torch.FloatTensor(metadata['vel_std']) ** 2 + vel_noise_std ** 2).to(device)

    pos_mean = torch.FloatTensor(metadata['pos_mean']).to(device)
    pos_std = torch.sqrt(torch.FloatTensor(metadata['pos_std']) ** 2 + pos_noise_std ** 2).to(device)
    
    # Normalization stats
    normalization_stats = {
        'acceleration': {'mean': acc_mean, 'std': acc_std},
        'velocity':    {'mean': vel_mean, 'std': vel_std},
        'position':     {'mean': pos_mean, 'std': pos_std}
    }
    
    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=FLAGS.dim,  # xyz
        nnode_in=(INPUT_SEQUENCE_LENGTH - 1) * FLAGS.dim + 16 + 8,  # timesteps * 3 (dim) + 16 (particle type embedding) + 5 meta features + 3 xyz
        nedge_in=FLAGS.dim + 1,    # input edge features, relative displacement in all dims + distance between two nodes
        latent_dim=FLAGS.hidden_dim,
        nmessage_passing_steps=FLAGS.layers,
        nmlp_layers=1,
        mlp_hidden_dim=FLAGS.hidden_dim,
        connectivity_radius=FLAGS.connection_radius,
        boundaries=np.array(metadata['bounds']),
        normalization_stats=normalization_stats,
        nparticle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
        device=device)

    return simulator


@torch.no_grad()
def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.tensor,
        particle_type: torch.tensor,
        n_particles_per_example: torch.tensor,
        strains: torch.tensor,
        nsteps: int,
        particle_dim: int,
        meta_feature: torch.tensor,
        device):
    """Rolls out a trajectory by applying the model recursively.
  
    Args:
      simulator: Learned simulator.
      features: Torch tensor features.
      nsteps: Number of steps.
    """
    # position is of shape [nparticles, timestep, dim], strains [timestep, nparticles]
    initial_positions = position
    initial_strains = strains
    
    current_positions = initial_positions
    pred_positions = []
    pred_strains = []
    
    boundary_mask = (particle_type == BOUNDARY_PARTICLE_ID).clone().detach().to(device)
    boundary_mask = boundary_mask.bool()[:, None].expand(-1, particle_dim)
    rebar_mask = (particle_type == REBAR_PARTICLE_ID).clone().detach().to(device)
    
    start_time = time.time()
    for step in range(nsteps):
        # Get next position with shape (nnodes, dim)
        next_position, pred_strain = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_type,
            meta_feature=meta_feature
        )
        
        next_position = torch.where(
            boundary_mask, initial_positions[:, 0], next_position)
        pred_strain = torch.where(
            boundary_mask[:, 0], initial_strains[0], pred_strain)

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1)
        
        pred_positions.append(next_position)
        pred_strains.append(pred_strain)
        
    run_time = time.time() - start_time
        
    # Predictions with shape (time, nnodes, dim)
    pred_positions = torch.stack(pred_positions)
    pred_strains = torch.stack(pred_strains)   
    
    initial_positions = initial_positions.permute(1, 0, 2)
    pred_trajs = torch.concatenate((initial_positions, pred_positions), axis=0)
    
    pred_strains[pred_strains < 0] = 0
    pred_strains = torch.concatenate((initial_strains, pred_strains), axis=0).cumsum(axis=0)
    pred_strains[pred_strains > 2] = 2
    
    output_dict = {
        'pred_trajs': pred_trajs.cpu().numpy(),
        'pred_strains': pred_strains.cpu().numpy(),
        'particle_type': particle_type.cpu().numpy(),
        'run_time': run_time
    }
            
    return output_dict


def load_sample(sample_path, metadata, device):
    STEP_SIZE = 6
    data = np.load(sample_path)
    # particle trajectory
    positions = data['particle_trajectories'].transpose((1, 0, 2)) # (nparticles, steps, 3)
    positions = positions[:, ::STEP_SIZE, :]
    positions = positions[:, -INPUT_SEQUENCE_LENGTH:]
    positions = torch.tensor(positions).to(torch.float32).contiguous().to(device)
    # particle type
    particle_type = data['particle_type']
    particle_type = torch.tensor(particle_type, dtype=torch.int32).contiguous().to(device)  
    n_particles_per_example = torch.tensor(positions.shape[0], dtype=torch.int32).to(device)
    # particle strain
    particle_strains = data['particle_strains']    
    #eps to epsi
    particle_strains = particle_strains[::STEP_SIZE, :]
    strains_diff = np.diff(particle_strains, axis=0)
    strains_diff[strains_diff < 0] = 0
    # 0-0.6 ms
    #particle_strains = np.concatenate((particle_strains[0:1, :], strains_diff), axis=0)
    # t + t+0.6 ms
    particle_strains = np.concatenate((particle_strains[-INPUT_SEQUENCE_LENGTH:-INPUT_SEQUENCE_LENGTH+1, :], strains_diff[-INPUT_SEQUENCE_LENGTH-1:]), axis=0)
                                                                                                                          
    particle_strains = particle_strains[:INPUT_SEQUENCE_LENGTH]
    
    particle_strains = torch.tensor(particle_strains).to(torch.float32).contiguous().to(device)
    # meta feature
    meta_feature = np.array([0, 0, 400, 0, 0, 5, 30])
    meta_feature[-2] = float(sample_path.split('.n')[0][-1])
    meta_feature = (meta_feature - metadata['meta_mean']) / metadata['meta_std']
    meta_feature = torch.tensor(meta_feature).to(torch.float32).to(device)
    
    return positions, particle_type, n_particles_per_example, particle_strains, meta_feature
    
def main(_):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"device = {device}")
    metadata = reading_utils.read_metadata(FLAGS.data_path)
    simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device)
    # Load weights
    try:
        simulator.load(FLAGS.model_path + FLAGS.model_file)
    except:
        print("Failed to load model weights!")
        sys.exit(1)
    simulator.to(device)
    simulator.eval()
    
    # Init output dir
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)
        
    # Load data
    # each step of FGN is 0.06 ms
    nsteps = 71
    dataset = glob.glob(osp.join(FLAGS.data_path, 'd3*.npz'))
    for idx, sample_path in enumerate(dataset):
        case_start_time = time.time()
        positions, particle_type, n_particles_per_example, strains, meta_feature = load_sample(sample_path,
                                                                                              metadata,
                                                                                              device)
        
        # Predict example rollout
        sample_output = rollout(simulator,
                                  positions,
                                  particle_type,
                                  n_particles_per_example,
                                  strains,
                                  nsteps,
                                  FLAGS.dim,
                                  meta_feature,
                                  device)
                             
        # # Save rollout in testing
        sample_output['metadata'] = metadata
        case_name = sample_path.split('.')[0].split('/')[-1] + '.pkl'
        filename = os.path.join(FLAGS.output_path, case_name)
        with open(filename, 'wb') as f:
            pickle.dump(sample_output, f)
        
        case_rollout_time = time.time() - case_start_time
        
        # Post-processing to extract reulst
        post_processing_start_time = time.time()
        post_processing.main('6', 
                             sample_output['pred_trajs'], 
                             sample_output['pred_strains'], 
                             sample_output['particle_type']
                            )
        postprocessing_time = time.time() - post_processing_start_time
        print(f"Finished {idx}/{len(dataset)} {case_name}, it takes {case_rollout_time:.0f}s and {postprocessing_time:.0f}s for rollout and post-processing on {device}")
        
if __name__ == '__main__':
    app.run(main)
