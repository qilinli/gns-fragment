
import numpy as np
import torch
import tree
import time
from absl import flags
from absl import app

from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader

INPUT_SEQUENCE_LENGTH = 3  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 2     # 0 beam, 1 load head, 2 long, 3 hoop, 4 broken particle
EROSIONAL_PARTICLE_ID = 10  # so broken particle will not contribute to loss


def rollout_rmse(pred, gt):
    """Rollout error in accumulated RMSE.
  
    Args:
        pred: prediction of shape [timesteps, nparticles, dim]
        gt: groundtruth of the same shape
    Return:
        loss: accumualted rmse loss of shape (timesteps,), where
        loss[t] is the average rmse loss of rolllout prediction of t steps
    """

    num_timesteps = gt.shape[0]
    squared_diff = np.square(pred - gt).reshape(num_timesteps, -1)
    loss = np.sqrt(np.cumsum(np.mean(squared_diff, axis=1), axis=0)/np.arange(1, num_timesteps+1))

    for show_step in range(0, num_timesteps, 3):
        if show_step < num_timesteps:
            print('Testing rmse  @ step %d loss: %.2e'%(show_step, loss[show_step]))
        else: break

    return loss


@torch.no_grad()
def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.tensor,
        particle_types: torch.tensor,
        n_particles_per_example: torch.tensor,
        strains: torch.tensor,
        nsteps: int,
        particle_dim: int,
        device):
    """Rolls out a trajectory by applying the model recursively.
  
    Args:
      simulator: Learned simulator.
      features: Torch tensor features.
      nsteps: Number of steps.
    """
    # position is of shape [nparticles, timestep, dim], strains [timestep, nparticles]
    initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
    initial_strains = strains[:INPUT_SEQUENCE_LENGTH,:]
    ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]
    ground_truth_strains = strains[INPUT_SEQUENCE_LENGTH:, :]
    nsteps = ground_truth_strains.shape[0]  # For 2D-T, nsteps vary between 29 and 30
    
    current_positions = initial_positions
    pred_positions = []
    pred_strains = []
    
    erosinal_mask = (particle_types == EROSIONAL_PARTICLE_ID).clone().detach().to(device)
    erosinal_mask = erosinal_mask.bool()[:, None].expand(-1, particle_dim)
    
    start_time = time.time()
    for step in range(nsteps):
        # Get next position with shape (nnodes, dim)
        next_position, pred_strain = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_types,
        )

        # Update erosinal particles from prescribed trajectory.
        next_position_ground_truth = ground_truth_positions[:, step]
        next_strain_ground_truth = ground_truth_strains[step, :]
        next_position = torch.where(
            erosinal_mask, next_position_ground_truth, next_position)
        pred_strain = torch.where(
            erosinal_mask[:, 0], next_strain_ground_truth, pred_strain)
        pred_positions.append(next_position)
        pred_strains.append(pred_strain)

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1)
        
        # Append gt to do 'one-step' prediction in rollout
        # current_positions = torch.cat(
        #     [current_positions[:, 1:], next_position_ground_truth[:, None, :]], dim=1)       
    
    run_time = time.time() - start_time
    
    # Predictions with shape (time, nnodes, dim)
    pred_positions = torch.stack(pred_positions)
    pred_strains = torch.stack(pred_strains)
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)
    
    # Note that this rmse loss is not comparable with training loss (MSE)
    # Besides, trainig loss is measured on acceleartion,
    # while rollout loss is on position
    rmse_position = rollout_rmse(pred_positions.cpu().numpy(), 
                                 ground_truth_positions.cpu().numpy()
                                )
    rmse_strain = rollout_rmse(pred_strains.cpu().numpy(), 
                               ground_truth_strains.cpu().numpy()
                              )  
    output_dict = {
        'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
        'initial_strains': initial_strains.cpu().numpy(),
        'predicted_rollout': pred_positions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'ground_truth_strain': ground_truth_strains.cpu().numpy(),
        'predicted_strain': pred_strains.cpu().numpy(),
        'particle_types': particle_types.cpu().numpy(),
        'rmse_position': rmse_position,
        'rmse_strain': rmse_strain,
        'run_time': run_time
    }

    return output_dict