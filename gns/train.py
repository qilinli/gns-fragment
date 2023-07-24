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
import wandb
import time

from absl import flags
from absl import app

from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import evaluate

import matplotlib.pyplot as plt


# Meta parameters
flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'], help=(
        'Train model, validation or rollout evaluation.'))
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')

# Model parameters
flags.DEFINE_float('connection_radius', 0.03, help='connectivity radius for graph.')
flags.DEFINE_integer('layers', 5, help='Number of GNN layers.')
flags.DEFINE_integer('hidden_dim', 64, help='Number of neurons in hidden layers.')
flags.DEFINE_integer('dim', 3, help='The dimension of concrete simulation.')

# Training parameters
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 6.7e-5, help='The std deviation of the noise.')
flags.DEFINE_integer('ntraining_steps', int(1E6), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# Continue training parameters
flags.DEFINE_string('model_file', None, help=(
    'Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=(
    'Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-3, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(4e5), help='Learning rate decay steps.')

# Wandb log parameters
flags.DEFINE_bool('log', False, help='if use wandb log.')
flags.DEFINE_string('project_name', 'GNS-tmp', help='project name for wandb log.')
flags.DEFINE_string('run_name', 'runrunrun', help='Run name for wandb log.')


FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 10  # So we can calculate the last 2 velocities.
NUM_PARTICLE_TYPES = 3     # concrete 0, rebar 1
REBAR_PARTICLE_ID = 1
BOUNDARY_PARTICLE_ID = 2


def predict(
        simulator: learned_simulator.LearnedSimulator,
        metadata: json,
        device: str):
    """Predict rollouts.
  
    Args:
      simulator: Trained simulator if not will exit.
      metadata: Metadata for test set.
  
    """
    # Load simulator
    try:
        simulator.load(FLAGS.model_path + FLAGS.model_file)
    except:
        print("Failed to load model weights!")
        sys.exit(1)

    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    # Use `valid`` set for eval mode if not use `test`
    split = 'test' if FLAGS.mode == 'rollout' else 'valid'

    data_trajs = data_loader.get_data_loader_by_trajectories(
        path=f"{FLAGS.data_path}{split}.npz")

    eval_loss = []
    with torch.no_grad():
        for example_i, data_traj in enumerate(data_trajs):
            nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
            positions = data_traj['positions'].to(device)
            particle_type = data_traj['particle_type'].to(device)
            strains = data_traj['strains'].to(device)
            meta_feature = data_traj['meta_feature'].to(device)
            
            # Predict example rollout
            example_output = evaluate.rollout(simulator,
                                              positions,
                                              particle_type,
                                              n_particles_per_example,
                                              strains,
                                              nsteps,
                                              FLAGS.dim,
                                              meta_feature,
                                              device)

            example_output['metadata'] = metadata

            # RMSE loss with shape (time,)
            loss_total = example_output['rmse_position'][-1] + example_output['rmse_strain'][-1]
            loss_position = example_output['rmse_position'][-1]
            loss_strain = example_output['rmse_strain'][-1]
            loss_oneStep = example_output['rmse_position'][0] + example_output['rmse_strain'][0]  

            print(f'''Predicting example {example_i}-
                  {example_output['metadata'][f'file_{split}'][example_i]} 
                  loss_toal: {loss_total}, 
                  loss_position: {loss_position}, 
                  loss_strain: {loss_strain}''')
            print(f"Prediction example {example_i} takes {example_output['run_time']}")
            eval_loss.append(loss_total)

            # Save rollout in testing
            if FLAGS.mode == 'rollout':
                example_output['metadata'] = metadata
                # simulation_name = metadata['file_test'][example_i]
                filename = f'rollout_{example_i}.pkl'
                filename = os.path.join(FLAGS.output_path, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(example_output, f)

    print("Mean loss on rollout prediction: {}".format(
        sum(eval_loss) / len(eval_loss)))
                  
                  
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
 
                        
def load_model(simulator, FLAGS, device):
    if FLAGS.model_file == "latest" and FLAGS.train_state_file == "latest":
        # find the latest model, assumes model and train_state files are in step.
        fnames = glob.glob(f"{FLAGS.model_path}*model*pt")
        max_model_number = 0
        expr = re.compile(".*model-(\d+).pt")
        for fname in fnames:
            model_num = int(expr.search(fname).groups()[0])
            if model_num > max_model_number:
                max_model_number = model_num
        # reset names to point to the latest.
        FLAGS.model_file = f"model-{max_model_number}.pt"
        FLAGS.train_state_file = f"train_state-{max_model_number}.pt"

    if os.path.exists(FLAGS.model_path + FLAGS.model_file) and os.path.exists(
        FLAGS.model_path + FLAGS.train_state_file):
        # load model
        simulator.load(FLAGS.model_path + FLAGS.model_file)

        # load train state
        train_state = torch.load(FLAGS.model_path + FLAGS.train_state_file)
        # set optimizer state
        optimizer = torch.optim.Adam(simulator.parameters())
        optimizer.load_state_dict(train_state["optimizer_state"])
        optimizer_to(optimizer, device)
        # set global train state
        step = train_state["global_train_state"].pop("step")

    else:
        msg = f'''Specified model_file {model_path + FLAGS.model_file}
        and train_state_file {model_path + FLAGS.train_state_file} not found.'''
        raise FileNotFoundError(msg)
    
    return simulator, step
    

def train(
        simulator: learned_simulator.LearnedSimulator,
        metadata: json,
        device: str):
    """Train the model.
  
    Args:
      simulator: Get LearnedSimulator.
    """
    optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init)
    step = 0
    # If model_path does not exist create new directory and begin training.
    model_path = FLAGS.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    # If model_path does exist and model_file and train_state_file exist continue training.
    if FLAGS.model_file is not None:
        simulator, step = load_model(simulator, FLAGS, device)
        
    simulator.train()
    simulator.to(device)

    data_samples = data_loader.get_data_loader_by_samples(path=f"{FLAGS.data_path}train.npz",
                                                          input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                          batch_size=FLAGS.batch_size,
                                                          )
    ### =================================== Training loop ============================================
    not_reached_nsteps = True
    lowest_eval_loss = 10000
    try:
        while not_reached_nsteps:
            for data_sample in data_samples:
                log = {}  # wandb logging
                # position are of size (nparticles*batch_size, INPUT_SEQUENCE_LENGTH, dim)             
                position = data_sample['input']['positions'].to(device)
                particle_type = data_sample['input']['particle_type'].to(device)
                n_particles_per_example = data_sample['input']['n_particles_per_example'].to(device)
                next_position = data_sample['output']['next_position'].to(device)
                next_strain = data_sample['output']['next_strain'].to(device)
                meta_feature = data_sample['meta']['meta_feature'].to(device)
                time_idx = data_sample['meta']['time_idx']


                # TODO (jpv): Move noise addition to data_loader
                # Sample the noise to add to the inputs to the model during training.
                sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position,
                                                                                        FLAGS.noise_std).to(device)
                non_boundary_mask = (particle_type != BOUNDARY_PARTICLE_ID).clone().detach().to(device)
                non_rebar_mask = (particle_type != REBAR_PARTICLE_ID).clone().detach().to(device)  
                sampled_noise *= non_boundary_mask.view(-1, 1, 1)

                # Get the predictions and target accelerations.
                pred_acc, target_acc, pred_strain = simulator.predict_accelerations(
                    next_positions=next_position,
                    position_sequence_noise=sampled_noise,
                    position_sequence=position,
                    nparticles_per_example=n_particles_per_example,
                    particle_types=particle_type,
                    meta_feature=meta_feature
                )
                
                ########## Debug
                # print('target acc mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(target_acc.mean().item(), 
                #                                                                          target_acc.std().item(),
                #                                                                          target_acc.min().item(),
                #                                                                          target_acc.max().item())
                #      )
                # print('pred acc mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(pred_acc.mean().item(), 
                #                                                                          pred_acc.std().item(),
                #                                                                          pred_acc.min().item(),
                #                                                                          pred_acc.max().item())
                #      )
                # if step % 10 == 0:
                #     np.save(f'debug/target_acc_{step:03}_{time_idx.item():03}', target_acc.detach().cpu().numpy())
                #     np.save(f'debug/pred_acc_{step:03}_{time_idx.item():03}', pred_acc.detach().cpu().numpy())
                # print('target strain', next_strain.mean().item(), next_strain.std().item())
                # print('pred strain', pred_strain.mean().item(), pred_strain.std().item())
                #############################
                
                ####### Calculate squared error
                # Compute acc loss only for non-boundary particles
                acc_mask = non_boundary_mask[:, None].expand(-1, target_acc.shape[-1])      # (nparticles, 3)
                squared_error_acc = (pred_acc - target_acc) ** 2                             # (nparticles, 3)
                squared_error_acc = torch.where(acc_mask.bool(),        
                                                squared_error_acc, 
                                                torch.zeros_like(squared_error_acc))         # (nparticles,3)
                mse_per_axis = squared_error_acc.mean(axis=0)                                # (3,), not proper mse, for log purpose only
                mse_acc = squared_error_acc.sum() / non_boundary_mask.sum()                 # scalar mse, x+y+z

                # Compute strain loss only for non-boundary and non-rebar partciels
                strain_mask = non_boundary_mask & non_rebar_mask                            # (nparticles,)
                squared_error_strain = (pred_strain - next_strain) ** 2                      # (nparticles,)
                squared_error_strain = torch.where(strain_mask.bool(),        
                                                   squared_error_strain, 
                                                   torch.zeros_like(squared_error_strain))   # (nparticles,)
                mse_strain = squared_error_strain.sum() / strain_mask.sum()                  # scalar mse
       
                # total loss
                loss = mse_acc + mse_strain

                # Computes the gradient of loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update learning rate
                lr_new = FLAGS.lr_init * (FLAGS.lr_decay ** (step / FLAGS.lr_decay_steps)) + 1e-6
                for param in optimizer.param_groups:
                    param['lr'] = lr_new
                
                # Print training info
                print("==Training step: {}/{}. Timestep: {:02}== "
                      "Total mse: {:8.6f},  Accleration mse: {:8.6f},  "
                      "Strain mse: {:8.6f}".format(
                          step, 
                          FLAGS.ntraining_steps, 
                          time_idx.item(),
                          loss.item(),
                          mse_acc.item(),
                          mse_strain.mean().item(),
                      ))
                
                # WandB logging
                log["train/mse-total"] = loss
                log["train/mse-strain"] = mse_strain
                log["train/mse-acc"] = mse_acc
                log["train/mse-x"] = mse_per_axis[0]
                log["train/mse-y"] = mse_per_axis[1]
                log["train/mse-z"] = mse_per_axis[2]
                log["lr"] = lr_new

                ### =================================== Validation rollout ============================================
                if step != 0 and step % FLAGS.nsave_steps == 0:
                    # Validation in the training loop
                    simulator.eval()
                    # Output path
                    if not os.path.exists(FLAGS.output_path):
                        os.makedirs(FLAGS.output_path)

                    # Use `valid`` set for eval mode if not use `test`
                    split = 'test' if FLAGS.mode == 'rollout' else 'valid'
                    data_trajs = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")

                    eval_loss_total, eval_loss_position, eval_loss_strain, eval_loss_oneStep = [], [], [], []
                    with torch.no_grad():
                        for example_i, data_traj in enumerate(data_trajs):
                            nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
                            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
                            positions = data_traj['positions'].to(device)
                            particle_type = data_traj['particle_type'].to(device)
                            strains = data_traj['strains'].to(device)
                            meta_feature = data_traj['meta_feature'].to(device)
                            
                            # Predict example rollout
                            example_output = evaluate.rollout(simulator,
                                                              positions,
                                                              particle_type,
                                                              n_particles_per_example,
                                                              strains,
                                                              nsteps,
                                                              FLAGS.dim,
                                                              meta_feature,
                                                              device)
                            
                            example_output['metadata'] = metadata

                            # RMSE loss with shape (time,)
                            loss_total = example_output['rmse_position'][-1] \
                                       + example_output['rmse_strain'][-1]
                            loss_position = example_output['rmse_position'][-1]
                            loss_strain = example_output['rmse_strain'][-1]
                            loss_oneStep = example_output['rmse_position'][0] ** 2 \
                                         + example_output['rmse_strain'][0] ** 2
                            
                            print(f'''Predicting example {example_i}-
                                  {example_output['metadata']['file_valid'][example_i]} 
                                  rmse_toal: {loss_total}, 
                                  rmse_position: {loss_position}, 
                                  rmse_strain: {loss_strain}''')
                            print(f"Prediction example {example_i} takes {example_output['run_time']}")
                            eval_loss_total.append(loss_total)
                            eval_loss_position.append(loss_position)
                            eval_loss_strain.append(loss_strain)
                            eval_loss_oneStep.append(loss_oneStep)
                        
                        eval_loss_mean = sum(eval_loss_total) / len(eval_loss_total)
                        print(f"Mean loss on valid-set rollout prediction: {eval_loss_mean}."
                              f"Current lowest eval loss is {lowest_eval_loss}.")
                        
                        # Save the current best model based on eval loss
                        if eval_loss_mean < lowest_eval_loss:
                            print(f"===================Better model obtained.=============================")
                            lowest_eval_loss = eval_loss_mean
                            if step > 0:
                                print(f"===================Saving.=============================")
                                save_dir = osp.join(model_path, FLAGS.run_name)
                                if not os.path.exists(save_dir):
                                    os.makedirs(save_dir)
                                simulator.save(osp.join(save_dir, f'model-{step:06}.pt'))
                                train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step": step})
                                #torch.save(train_state, osp.join(save_dir, f'train_state-{step:06}.pt'))

                        # log
                        log["val/rmse-total"] = sum(eval_loss_total) / len(eval_loss_total)
                        log["val/rmse-position"] = sum(eval_loss_position) / len(eval_loss_position)
                        log["val/rmse-strain"] = sum(eval_loss_strain) / len(eval_loss_strain)
                        log["val/mse-oneStep"] = sum(eval_loss_oneStep) / len(eval_loss_oneStep)
                    # =========================================================

                # Complete training
                if (step >= FLAGS.ntraining_steps):
                    not_reached_nsteps = False
                    break

                if FLAGS.log:  
                    wandb.log(log, step=step)
                    
                step += 1

    except KeyboardInterrupt:
        pass
    
    save_dir = osp.join(model_path, FLAGS.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                                           
    simulator.save(osp.join(save_dir, f'model-{step:06}.pt'))
    train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step": step})
    torch.save(train_state, osp.join(save_dir, f'train_state-{step:06}.pt'))


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
    pos_noise_std = torch.FloatTensor([0.005, 0.005, 0.1])
    
    
    acc_mean = torch.FloatTensor(metadata['acc_mean']).to(device)
    acc_std = torch.sqrt(torch.FloatTensor(metadata['acc_std']) ** 2 + acc_noise_std ** 2).to(device)
    # acc_std = torch.FloatTensor(metadata['acc_std'])
    # acc_std = torch.sqrt(acc_std ** 2 + (acc_std / FLAGS.noise_std) ** 2).to(device)
    
    vel_mean = torch.FloatTensor(metadata['vel_mean']).to(device)
    vel_std = torch.sqrt(torch.FloatTensor(metadata['vel_std']) ** 2 + vel_noise_std ** 2).to(device)
    # vel_std = torch.FloatTensor(metadata['vel_std'])
    # vel_std = torch.sqrt(vel_std ** 2 + (vel_std / FLAGS.noise_std) ** 2).to(device)
    
    pos_mean = torch.FloatTensor(metadata['pos_mean']).to(device)
    pos_std = torch.sqrt(torch.FloatTensor(metadata['pos_std']) ** 2 + pos_noise_std ** 2).to(device)
    
    # Normalization stats
    normalization_stats = {
        'acceleration': {'mean': acc_mean, 'std': acc_std},
        'velocity':    {'mean': vel_mean, 'std': vel_std},
        'position':     {'mean': pos_mean, 'std': pos_std}
    }
    
    # num_boundary_fea = 4 if FLAGS.dim == '1d' else 4 #qilin
    
    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=FLAGS.dim,  # xyz
        nnode_in=(INPUT_SEQUENCE_LENGTH - 1) * FLAGS.dim + 16 + 8,  # timesteps * 3 (dim) + 9 (particle type embedding) + 5 meta features
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


def main(_):
    """Train or evaluates the model.
  
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # Read metadata
    metadata = reading_utils.read_metadata(FLAGS.data_path)
    simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device)

    if FLAGS.mode == 'train':
        # Init wandb
        if FLAGS.log:
            wandb.init(project=FLAGS.project_name, name=FLAGS.run_name)
            train(simulator, metadata, device)
            wandb.finish()
        else:
            train(simulator, metadata, device)

    elif FLAGS.mode in ['valid', 'rollout']:
        predict(simulator, metadata, device)


if __name__ == '__main__':
    app.run(main)
